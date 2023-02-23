import xarray as xr
import numpy as np
import configparser
import os
import glob
import re
from datetime import datetime
import copy 
import argparse
import warnings
import xesmf as xe
import random
from collections import OrderedDict
from tensorflow.keras.utils import to_categorical
import netCDF4 as nc
from tqdm import tqdm
import time
import importlib
from helper_functions import Kernelizer, setup_valid_pixel_array, get_indices_of_dims_to_collapse


# NOTE: netcdf4 also need to be manually installed as dependencies
#       IMPORTANT: First thing to install on new env is xesmf/esmpy using the command conda install -c conda-forge xesmf esmpy=8.0.0
#       this will install python as well. Second thing to install is cftime. Third thing is to install dask. Then see netcdf4 above.  
#       tensorflow must also be version 2.7.0 or greater (if using keras-unet-collection in another script).

# NOTE: Spatial dataset delineation in file/path naming not currently supported.


class TimedeltaPair:
    # Takes in numpy.timedelta64 objects for both arguments
    def __init__(self, val_timedelta=None, init_timedelta=None):
        self.timedelta_pair = (val_timedelta, init_timedelta)
        if init_timedelta is None:
            self.has_init_timedelta = False
        else:
            if type(init_timedelta) is not np.timedelta64:
                raise Exception("Only numpy.timedelta64 objects are accepted in this constructor.")
            self.has_init_timedelta = True
        if val_timedelta is None:
            self.has_val_timedelta = False
        else:
            if type(val_timedelta) is not np.timedelta64:
                raise Exception("Only numpy.timedelta64 objects are accepted in this constructor.")
            self.has_val_timedelta = True


class DatetimePair:
    # Takes in numpy.datetime64 objects for both datetime arguments
    # The res arguments are for determining if the numpy datetime objects need to be changed so their dates are actually comparing apples to apples
    # if their resolution is different. (Ex: comparing datetimes for hourly data to data that occurs every 5 minutes). Only used in the __eq__ method.
    def __init__(self, val_datetime, val_res_str, val_res_int, init_datetime=None, init_res_str=None, init_res_int=None):
        self.datetime_pair = (val_datetime, init_datetime)
        self.val_res_str = val_res_str
        self.val_res_int = val_res_int

        if type(val_datetime) is not np.datetime64:
            raise Exception("Only numpy.datetime64 objects are accepted in this constructor.")

        if init_datetime is None:
            self.has_init_datetime = False
            self.init_res_str = None
            self.init_res_int = None

        else:
            if type(init_datetime) is not np.datetime64:
                raise Exception("Only numpy.datetime64 objects are accepted in this constructor.")
            if init_res_str is None or init_res_int is None:
                raise Exception("If giving init datetimes, you must also supply the init resolution parameters.")
            self.has_init_datetime = True
            self.init_res_str = init_res_str
            self.init_res_int = init_res_int

    def __eq__(self, other):
        if type(other) is not DatetimePair:
            raise Exception("Only a DatetimePair object is a valid input for this operator.")

        self_val_datetime = self.datetime_pair[0]
        other_val_datetime = other.datetime_pair[0]
        if self.val_res_int < other.val_res_int:
            other_val_datetime = other.datetime_pair[0].astype(self.val_res_str)
        else:
            self_val_datetime = self.datetime_pair[0].astype(other.val_res_str)

        if self.has_init_datetime and other.has_init_datetime:
            self_init_datetime = self.datetime_pair[1]
            other_init_datetime = other.datetime_pair[1]
            if self.init_res_int < other.init_res_int:
                other_init_datetime = other.datetime_pair[1].astype(self.init_res_str)
            else:
                self_init_datetime = self.datetime_pair[1].astype(other.init_res_str)

            return self_val_datetime == other_val_datetime and self_init_datetime == other_init_datetime
        else:
            return self_val_datetime == other_val_datetime

    # Expects either DatetimePair or TimedeltaPair
    def __add__(self, other):
        if type(other) is DatetimePair:
            if self.has_init_datetime and other.has_init_datetime:
                init_timedelta = self.datetime_pair[1] + other.datetime_pair[1]
            else:
                init_timedelta = None
            val_timedelta = self.datetime_pair[0] + other.datetime_pair[0]
            return TimedeltaPair(val_timedelta=val_timedelta, init_timedelta=init_timedelta)

        elif type(other) is TimedeltaPair:
            if self.has_init_datetime and other.has_init_timedelta:
                init_datetime = self.datetime_pair[1] + other.timedelta_pair[1]
            else:
                init_datetime = self.datetime_pair[1]
            if other.has_val_timedelta:
                val_datetime = self.datetime_pair[0] + other.timedelta_pair[0]
            else:
                val_datetime = self.datetime_pair[0]
            return DatetimePair(val_datetime, self.val_res_str, self.val_res_int, init_datetime, self.init_res_str, self.init_res_int)

        else:
            raise Exception("Only DatetimePair or TimedeltaPair are valid inputs for this operator.")

    def __sub__(self, other):
        if type(other) is DatetimePair:
            if self.has_init_datetime and other.has_init_datetime:
                init_timedelta = self.datetime_pair[1] - other.datetime_pair[1]
            else:
                init_timedelta = None
            val_timedelta = self.datetime_pair[0] - other.datetime_pair[0]
            return TimedeltaPair(val_timedelta=val_timedelta, init_timedelta=init_timedelta)

        elif type(other) is TimedeltaPair:
            if self.has_init_datetime and other.has_init_timedelta:
                init_datetime = self.datetime_pair[1] - other.timedelta_pair[1]
            else:
                init_datetime = self.datetime_pair[1]
            if other.has_val_timedelta:
                val_datetime = self.datetime_pair[0] - other.timedelta_pair[0]
            else:
                val_datetime = self.datetime_pair[0]
            return DatetimePair(val_datetime, self.val_res_str, self.val_res_int, init_datetime, self.init_res_str, self.init_res_int)

        else:
            raise Exception("Only DatetimePair or TimedeltaPair are valid inputs for this operator.")

    def __str__(self):
        return str(self.datetime_pair)

    def __repr__(self):
        return self.__str__()
    
    def val_time_equals_init_time(self):
        return self.datetime_pair[0] == self.datetime_pair[1]


class Patcher:
    def __init__(self, run_num, config_path):
        # Parse in config file specified by config_path. See examples given in repo
        config = configparser.ConfigParser()
        config.read(config_path)
        settings_dict = cfg_parser(config)
        self.run_num = run_num

        # Create all top level settings dicts
        self.top_settings_patches = settings_dict["Patches"]
        self.top_settings_input = settings_dict["Input_Data"]
        self.top_settings_output = settings_dict["Output"]
        self.top_settings_stopping = settings_dict["Stopping"]

        # Throw exception if we do not have enough dataset names in top config
        if len(self.top_settings_input["input_cfg_list"]) != len(self.top_settings_input["dataset_names"]):
            raise Exception('Number of elements in "dataset_names" does not equal number of elements in "input_cfg_list"!')

        # Create all data settings configs
        self.data_settings = []
        for i, data_settings_cfg in enumerate(self.top_settings_input["input_cfg_list"]):
            config = configparser.ConfigParser()
            config.read(data_settings_cfg)
            data_setting = cfg_parser(config)
            data_setting["dataset_name"] = self.top_settings_input["dataset_names"][i]
            self.data_settings.append(data_setting)

        # Load the rest of the global variables/constants that I use throughout the object
        self._load_all_dataset_metadata()

    
    def offset_datasets_and_make_governing_dataset(self):
        new_datasets_paths = []
        new_datasets_datetimes = []
        new_dataset_pos_in_file = []
        new_dataset_full_list_pos = []
        new_data_settings = []
        self.non_offset_datetimes = []

        # Find out if at least one of the datasets we are using has init times. This is for an exception in the next block.
        has_init_time = False
        for data_settings_cfg in self.data_settings:
            if data_settings_cfg["Path"]["init_dt_positions"] is not None and data_settings_cfg["Path"]["init_dt_regs"] is not None and data_settings_cfg["Path"]["init_dt_formats"] is not None:
                has_init_time = True

        self.gov_datetimes = None

        for i, dataset_datetimes in enumerate(self.datasets_datetimes):
            data_settings_cfg = self.data_settings[i]

            time_offsets = data_settings_cfg["Data"]["time_offsets"]
            if len(time_offsets) != 0:
                for time_offset in time_offsets:
                    time_offset_np = np.timedelta64(time_offset, 'm')
                    new_datasets_datetimes.append(dataset_datetimes - TimedeltaPair(val_timedelta=time_offset_np))
                    new_datasets_paths.append(copy.deepcopy(self.datasets_paths[i]))
                    new_dataset_pos_in_file.append(copy.deepcopy(self.dataset_pos_in_file[i]))
                    new_dataset_full_list_pos.append(copy.deepcopy(self.dataset_full_list_pos[i]))
                    new_data_settings.append(copy.deepcopy(data_settings_cfg))
                    self.non_offset_datetimes.append(copy.deepcopy(dataset_datetimes))
            else:
                new_datasets_datetimes.append(copy.deepcopy(dataset_datetimes))
                new_datasets_paths.append(copy.deepcopy(self.datasets_paths[i]))
                new_dataset_pos_in_file.append(copy.deepcopy(self.dataset_pos_in_file[i]))
                new_dataset_full_list_pos.append(copy.deepcopy(self.dataset_full_list_pos[i]))
                new_data_settings.append(copy.deepcopy(data_settings_cfg))
                self.non_offset_datetimes.append(copy.deepcopy(dataset_datetimes))

            # Setup governing dataset that controls the flow of all searches
            if data_settings_cfg["Data"]["govern_search"]:
                # Raise an exception if the chosen governing dataset does not have init time when other datasets do have it.
                if (data_settings_cfg["Path"]["init_dt_positions"] is None or data_settings_cfg["Path"]["init_dt_regs"] is None or data_settings_cfg["Path"]["init_dt_formats"] is None) and has_init_time:
                    raise Exception('At least one of your chosen datasets uses initialization time, however the dataset you selected with "govern_search" does not have initialization time. This cannot be! If you are working with initialization time, please make sure the governing dataset has initialization time!')
                
                if self.top_settings_input["anchor_offsets_to_init"]:
                    self.gov_datetimes = []
                    for dataset_datetime in new_datasets_datetimes[-1]:
                        if dataset_datetime.val_time_equals_init_time():
                            self.gov_datetimes.append(dataset_datetime)
                    self.gov_datetimes = np.array(self.gov_datetimes)
                else:
                    self.gov_datetimes = new_datasets_datetimes[-1]

        if self.gov_datetimes is None:
            raise Exception('No dataset selected to be governing dataset with "govern_search". Please run again with one dataset selected as the governing dataset!')
        
        print("Governing dataset size: " + str(len(self.gov_datetimes)))
        
        self.datasets_paths = new_datasets_paths
        self.datasets_datetimes = new_datasets_datetimes
        self.dataset_pos_in_file = new_dataset_pos_in_file
        self.dataset_full_list_pos = new_dataset_full_list_pos
        self.data_settings = new_data_settings


    # If doing parallel runs on supercomputer, split our data across the runs so there is no double sampling of patches
    def _split_datasets_for_parallel_jobs(self):
        n_parallel_runs = self.top_settings_patches["n_parallel_runs"]
        
        if n_parallel_runs is not None and n_parallel_runs != 0:
            gov_datetimes_split = np.array_split(self.gov_datetimes, n_parallel_runs)
            self.gov_datetimes = gov_datetimes_split[self.run_num]

            print("Governing dataset's date count after parallel job split: " + str(len(self.gov_datetimes)))

    
    def _load_all_dataset_metadata(self):
        data_start = self.top_settings_input["data_start"]
        data_end = self.top_settings_input["data_end"]

        filtered_balanced_counts = []
        self.dataset_start_times = []
        self.dataset_end_times = []
        # This loop sets up the start and end dates for the filtering done in self.get_files_and_datetimes(). It needs to be a loop
        # becuase it checks if start and ends times set in the top config (low priority) or the dataset configs (high priority)
        for data_settings_cfg in self.data_settings:
            data_start_for_one_ds = copy.deepcopy(data_start)
            data_end_for_one_ds = copy.deepcopy(data_end)
            if data_settings_cfg["Bounds"]["data_start"] is not None or data_settings_cfg["Bounds"]["data_end"] is not None:
                data_start_for_one_ds = data_settings_cfg["Bounds"]["data_start"]
                data_end_for_one_ds = data_settings_cfg["Bounds"]["data_end"]

            self.dataset_start_times.append(data_start_for_one_ds)
            self.dataset_end_times.append(data_end_for_one_ds)
            
            if data_start_for_one_ds is None and data_end_for_one_ds is None:
                raise Exception("You must give at least some data bounds in either the top config or in the dataset configs.")  

        for i in self.top_settings_patches["filters_balanced"]:
            filtered_balanced_counts.append(0)
        if len(filtered_balanced_counts) == 0:
            filtered_balanced_counts = [0]

        # Initialize ints that my netcdf loader needs to run optimally. Done here so netcdf load calls in dataset_netcdf_load_modes work alright
        # but this has to be initialized again below once correct dataset sizes done.
        self.dataset_netcdf_load_modes = np.zeros(len(self.data_settings), dtype=np.int64)

        # Load all file paths accordiong to given regex, get datetimes according to regex, and perform datetime filtering.
        self.get_files_and_datetimes()

        # Print a summary of found datetimes across all datasets
        file_counts = [len(datasets_datetime) for datasets_datetime in self.datasets_datetimes]
        print('Number of dates found: ' + str(file_counts))

        # Make new datasets (paths, datetimes, index information, etc.) with an offset time so that consecutive times can be matched
        # to make pathces with an additional time dimension. Also make the governing dataset which is a just a list of datetimes that we match
        # all dataset's datetime to. This system forms the basis of how we search for and find matching data across all datasets.
        self.offset_datasets_and_make_governing_dataset()

        # Initialize ints that my netcdf loader needs to run optimally. Reset here becuase datasets are now at correct size
        self.dataset_netcdf_load_modes = np.zeros(len(self.data_settings), dtype=np.int64)

        # If doing parallel runs on supercomputer, split our data across the runs so there is no double sampling of patches
        self._split_datasets_for_parallel_jobs()

        # Set top level counters
        self.date_counter = 0
        self.filtered_balanced_counts = filtered_balanced_counts

    
    def run(self):
        feature_patches_root = self.top_settings_output["examples_root"]
        label_patches_root = self.top_settings_output["labels_root"]
        n_patches = self.top_settings_patches["number_of_patches"]
        max_num_of_searches = self.top_settings_stopping["max_num_of_searches"]

        self.patch_size = self.top_settings_patches["patch_size"]
        self.patches_per_time = self.top_settings_patches["patches_per_unit_time"]
        self.shuffle_patches_in_each_timestep = self.top_settings_patches["shuffle_patches_in_each_timestep"]
        self.ignore_nans = self.top_settings_patches["ignore_nans"]
        
        main_loop_counter = 0
        
        #TODO: Make a bunch of checks here (prob in another method) that will throw full exceptions if stuff missing that is needed for the main loop.
        # For example check if the dataset(s) are all empty

        self.number_of_patches_per_balanced_var = n_patches / len(self.filtered_balanced_counts)
        self.feature_patches = None
        self.label_patches = None

        while np.any(np.array(self.filtered_balanced_counts) < self.number_of_patches_per_balanced_var):
            print("---------------------------------------------")
            if main_loop_counter % 5 == 0:
                print("Reached search number: " + str(main_loop_counter))

            if main_loop_counter == max_num_of_searches:
                warnings.warn('WARNING: Hit maximum number of allowed searches set by "max_num_of_searches". Number of completed patches may be less than expected.')
                break
            main_loop_counter = main_loop_counter + 1

            start_time = time.time()
            found_files = self._find_indeces_of_matching_datasets()
            print("Matching dataset time: " + str(time.time() - start_time))

            if not found_files:
                warnings.warn('WARNING: Ran out of files with matching datetimes. Number of completed patches may be less than expected. Please consider adjusting "patches_per_unit_time"')
                break

            self._load_datasets_from_disk()

            if self.dataset_empty_or_out_of_range:
                warnings.warn('WARNING: At least one of the selected dataset files contained data that was entirely missing or data that did not spatially align with the other datasets. Continuing search...')
                continue
            if self.master_xarray_dataset_incompatibility:
                warnings.warn('WARNING: When creating the master xarray dataset a merge conflict occured. Maybe there are some erroneous latlons? Continuing search...')
                continue

            patch_count_last_search = np.sum(self.filtered_balanced_counts)

            self._make_patches()

            # Print the indices in n_samples for the patches added in this search
            patch_count_this_search = np.sum(self.filtered_balanced_counts)
            if patch_count_this_search - patch_count_last_search == 0:
                indices_str = "NONE"
            else:
                indices_str = str(np.arange(patch_count_last_search, patch_count_this_search).tolist())
            print("Indices of patches added in this search: " + indices_str)
            print("---------------------------------------------")

        start_time = time.time()
        if self.feature_patches is not None:
            feature_patch_path = os.path.join(feature_patches_root, "{:04d}".format(self.run_num) + ".nc")
            for var_key in list(self.feature_patches.keys()):
                if "units" in self.feature_patches[var_key].attrs:
                    if type(self.feature_patches[var_key].attrs["units"]) is not str:
                        self.feature_patches[var_key].attrs["units"] = ""
            self.feature_patches.attrs = {}
            self.feature_patches.to_netcdf(feature_patch_path)
        print("Save examples netcdf time: " + str(time.time() - start_time))
        
        start_time = time.time()
        if self.label_patches is not None:
            label_patch_path = os.path.join(label_patches_root, "{:04d}".format(self.run_num) + ".nc")
            for var_key in list(self.label_patches.keys()):
                if "units" in self.label_patches[var_key].attrs:
                    if type(self.label_patches[var_key].attrs["units"]) is not str:
                        self.label_patches[var_key].attrs["units"] = ""
            self.label_patches.attrs = {}
            self.label_patches.to_netcdf(label_patch_path)
        print("Save labels netcdf time: " + str(time.time() - start_time))

        print("Completed on search number: " + str(main_loop_counter))
        print("Final patch count per balanced class:")
        print(self.filtered_balanced_counts)


    def make_datetime_netcdf_file(self, single_dataset_paths, dataset_index, lat_dim, lon_dim, time_cord_name, time_dim_name, output_path):
        warnings.warn("Starting internal datetime saving process, this will take a while. This is needed if you are working with datasets that have internal times. File only has to be made once.")

        if self.top_settings_patches["n_parallel_runs"] != 1 or self.run_num != 0:
            raise Exception("When making a datetime netcdf file you MUST have only one proccess running and it must be labeled as run number 0.")

        extracted_ds = None
        for i in tqdm(range(len(single_dataset_paths))):
            # Load one netcdf
            ds = self._netcdf_loader(single_dataset_paths[i], dataset_index, lat_dim, lon_dim)

            # Assign indices for both the position of the chosen file in its original list and also the position of each datetime within the file
            ds = ds.assign(full_list_pos=([time_dim_name], np.ones(len(ds[time_cord_name]), dtype=np.int64)*i))
            ds = ds.assign(pos_in_file=([time_dim_name], np.arange(len(ds[time_cord_name]))))

            # Select only the 3 wanted variables
            ds = ds[[time_cord_name, 'full_list_pos', 'pos_in_file']]

            # This block is for concatenating the individual datasets together into one xarray dataset
            if extracted_ds is None:
                extracted_ds = copy.deepcopy(ds)
            else:
                extracted_ds = xr.concat([extracted_ds, ds], dim=time_dim_name)
            
            ds.close()

        if extracted_ds is not None:
            extracted_ds.to_netcdf(output_path)

        quit()

            
    def load_datetimes_from_pre_saved_netcdf(self, path, time_cord_name):
        ds = xr.open_dataset(path)

        ds_datetimes = ds[time_cord_name].to_numpy()
        ds_full_pos = ds["full_list_pos"].to_numpy()
        ds_pos_in_file = ds["pos_in_file"].to_numpy()

        ds.close()

        return ds_datetimes, ds_full_pos, ds_pos_in_file

    
    def view_dataset(self, ds_path, lat_dim = None, lon_dim = None):
        try:
            self.dataset_netcdf_load_modes = np.array([0])
            ds = self._netcdf_loader(ds_path, 0, lat_dim, lon_dim)
        except:
            raise Exception("File loading failed. If you haven't already try to include the lat and lon dims of the dataset if you know them and try again.")

        ds_keys = [key for key in ds.keys()]

        print("---------------------------")
        print("DATASET XARRAY SUMMARY:")
        print("---------------------------")
        print(ds)

        print("---------------------------")
        print("DATASET VARS AND/OR COORDS KEY(s):")
        print("---------------------------")
        print(ds_keys)


    # TODO: MAJOR: Make exception in here for case where decode_cf=False has to run but the dataset's has_time_cord=True
    # TODO: Consider if the drop_variables stuff with the lat and lon dims should be the lat lon cords instead. How can you drop an entire dim?
    def _netcdf_loader(self, path, dataset_index, lat_dim = None, lon_dim = None):
        need_to_rename_dims = False
        if self.dataset_netcdf_load_modes[dataset_index] == 1:
            ds = xr.open_dataset(path, decode_cf=False)
        elif self.dataset_netcdf_load_modes[dataset_index] == 2:
            if lat_dim is None or lon_dim is None:
                raise Exception("This dataset requires that we rename it's dimensions. You must not keep lat_dim and lon_dim as None.")
            
            ds = xr.open_dataset(path, decode_cf=False, drop_variables=[lat_dim, lon_dim])
            need_to_rename_dims = True
        else:
            try:
                ds = xr.open_dataset(path, decode_cf=True)
            except:
                warnings.warn('WARNING: One or more of your selected dataset(s) contains at least one netcdf file that does not follow netcdf convections. Less useful netcdf loader had to be used.')
                try:
                    ds = xr.open_dataset(path, decode_cf=False)
                    self.dataset_netcdf_load_modes[dataset_index] = 1
                except:
                    try:
                        if lat_dim is None or lon_dim is None:
                            raise Exception("")
                        
                        ds = xr.open_dataset(path, decode_cf=False, drop_variables=[lat_dim, lon_dim])
                        need_to_rename_dims = True
                        self.dataset_netcdf_load_modes[dataset_index] = 2
                    except:
                        raise Exception('Unable to load at least one of the netcdf files! Had to kill patcher. Note: This may mean I need a more robust _netcdf_loader method. Talk to Tobias.')

        # This is to deal with case where the new lon/lat coordinates we need to make have the same dimension names. Xarray doesn't like that
        # NOTE: This changes the dims without changing the dataset's y_dim_name or x_dim_name setting in the setting dict
        dims = [dim for dim in ds.dims]
        if "lon" in dims or need_to_rename_dims:
            if lon_dim is None:
                raise Exception("This dataset requires that we rename it's dimensions. You must not keep lat_dim and lon_dim as None.")
            ds = ds.rename_dims({lon_dim: "lon_dim"})
        if "lat" in dims or need_to_rename_dims:
            if lat_dim is None:
                raise Exception("This dataset requires that we rename it's dimensions. You must not keep lat_dim and lon_dim as None.")
            ds = ds.rename_dims({lat_dim: "lat_dim"})

        return ds


    def _find_indeces_of_matching_datasets(self):
        found_files = False

        while not found_files and self.date_counter < len(self.gov_datetimes):
            self.date_indices = []
            self.time_indices = []
            self.time_str_indices = [] # For the printing of the datetimes in the load_dataset_from_disk method

            for i, dataset_datetimes in enumerate(self.datasets_datetimes):
                current_datetime = self.gov_datetimes[self.date_counter]

                date_index = np.nonzero(np.equal(current_datetime, dataset_datetimes))

                if len(date_index[0]) == 0:
                    break
                
                date_index = date_index[0][0]
                
                self.time_indices.append(self.dataset_pos_in_file[i][date_index])
                self.date_indices.append(self.dataset_full_list_pos[i][date_index])
                self.time_str_indices.append(date_index)

            if len(self.date_indices) == len(self.datasets_datetimes):
                found_files = True
            
            self.date_counter = self.date_counter + 1

        return found_files

    
    # Add any new cariables to the dataset. post_reproj is a flag for determinging if this new variable is meant to be 
    # added before or after the reprojection step.
    def _add_custom_vars(self, ds, data_settings_cfgs, current_ds_index, post_reproj):
        if post_reproj:
            setting_name = "post_reproj_custom_vars"
        else:
            setting_name = "custom_vars"

        for j, custom_var in enumerate(data_settings_cfgs[current_ds_index]["Modification"][setting_name]):
            return_dict = OrderedDict()
            exec(custom_var, globals().update(locals()), return_dict)
            return_list = list(return_dict.items())
            var_name = return_list[-1][0]
            values = return_list[-1][-1]
            ds = ds.assign({var_name: values})
            
            n_classes = data_settings_cfgs[current_ds_index]["Modification"]["n_classes_for_new_vars"][j]
            if n_classes >= 2:
                all_keys = [key for key in ds.keys()]
                previous_key = all_keys[-1]
                new_var = ds[previous_key]
                new_var_dim_names = new_var.dims

                new_var_cat = to_categorical(new_var, num_classes=n_classes)

                for i in range(new_var_cat.shape[-1]):
                    new_var_class_i = new_var_cat[...,i]
                    new_var_name_i = previous_key + "_" + str(i)
                    ds = ds.assign({new_var_name_i: (new_var_dim_names, new_var_class_i)})
                
                ds = ds.drop(previous_key)

        return ds

    
    def _select_specific_dims(self, ds, data_settings_cfgs, current_ds_index):
        for i, dim_selection_name in enumerate(data_settings_cfgs[current_ds_index]["Filtration"]["dim_selection_names"]):
            dim_selection_index = data_settings_cfgs[current_ds_index]["Filtration"]["dim_selection_index"][i]
            if type(dim_selection_index) == int:
                ds = ds[{dim_selection_name: dim_selection_index}]
            elif dim_selection_index == "*":
                random_index = random.randint(0,len(ds[dim_selection_name].to_numpy())-1)
                ds = ds[{dim_selection_name: random_index}]
            else:
                raise Exception('Invalid input received for dim_selection_index setting. Must be int or "*".')
        
        return ds

    
    def _load_datasets_from_disk(self): #, chosen_date_indeces, all_found_time_indeces_adjusted, solution_indeces_files, solution_indeces_times, datasets_paths, data_settings_cfgs):
        loaded_datetimes = []
        loaded_filenames = []
        loaded_datasets = []
        loaded_datasets_examples = []
        loaded_datasets_labels = []
        loaded_datetimes_examples = []
        loaded_datetimes_labels = []
        dataset_configs_examples = []
        dataset_configs_labels = []
        adjusted_x_dim_names_examples = []
        adjusted_y_dim_names_examples = []
        adjusted_x_dim_names_labels = []
        adjusted_y_dim_names_labels = []
        data_settings_cfgs = self.data_settings

        start_time = time.time()
        for i, (file_index, time_index, time_str_index) in enumerate(zip(self.date_indices, self.time_indices, self.time_str_indices)):
            loaded_datetimes.append(self.non_offset_datetimes[i][time_str_index])
            loaded_filenames.append(self.datasets_paths[i][file_index])

            path = self.datasets_paths[i][file_index]
            
            y_dim_name = data_settings_cfgs[i]["Data"]["y_dim_name"]
            x_dim_name = data_settings_cfgs[i]["Data"]["x_dim_name"]
            ds = self._netcdf_loader(path, i, y_dim_name, x_dim_name)
            nc_ds = nc.Dataset(path)
            lats = nc_ds[data_settings_cfgs[i]["Data"]["lat_cord_name"]][:]
            lons = nc_ds[data_settings_cfgs[i]["Data"]["lon_cord_name"]][:]
            nc_ds.close()

            if data_settings_cfgs[i]["Data"]["has_time_cord"]:
                time_dim_name = data_settings_cfgs[i]["Data"]["time_dim_name"]
                ds = ds[{time_dim_name: time_index}]

            if len(lons.shape) == 1:
                lons, lats = np.meshgrid(lons, lats)
            elif len(lons.shape) == 2:
                pass
            else:
                raise Exception("At least one dataset has lat/lons with either too few or too many dimensions. lat/lons must be either 2d or 1d.")
            
            dims = [dim for dim in ds.dims]
            if "lon_dim" not in dims or "lat_dim" not in dims:
                ds = ds.rename_dims({data_settings_cfgs[i]["Data"]["x_dim_name"]: "lon_dim"})
                ds = ds.rename_dims({data_settings_cfgs[i]["Data"]["y_dim_name"]: "lat_dim"})
            y_dim_name = "lat_dim"
            x_dim_name = "lon_dim"

            # Select only the data we want
            ds = ds[data_settings_cfgs[i]["Data"]["selected_vars"]]

            # Set to float32 if needed:
            if self.top_settings_patches["make_float32"]:
                ds = ds.astype(np.float32)

            # If only one key is given ds is reverted to a xarray dataarray.
            # This is not useful for later functions so this is changed back to a xarray dataset
            if type(ds) is xr.core.dataarray.DataArray:
                ds = ds.to_dataset()

            ds = ds.assign_coords(lon=((y_dim_name,x_dim_name), lons))
            ds = ds.assign_coords(lat=((y_dim_name,x_dim_name), lats))

            ds = self._select_specific_dims(ds, data_settings_cfgs, i)
            ds = self._add_custom_vars(ds, data_settings_cfgs, i, False)

            if data_settings_cfgs[i]["Data"]["is_label_data"]:
                loaded_datetimes_labels.append(self.non_offset_datetimes[i][time_str_index])
                adjusted_x_dim_names_labels.append(x_dim_name)
                adjusted_y_dim_names_labels.append(y_dim_name)
                dataset_configs_labels.append(data_settings_cfgs[i])
            else:
                loaded_datetimes_examples.append(self.non_offset_datetimes[i][time_str_index])
                adjusted_x_dim_names_examples.append(x_dim_name)
                adjusted_y_dim_names_examples.append(y_dim_name)
                dataset_configs_examples.append(data_settings_cfgs[i])
            
            loaded_datasets.append(ds)

        print("Load dataset time: " + str(time.time() - start_time))
        
        # Print to stdout information about this search
        if self.top_settings_patches["run_debug_text"]:
            print("File(s) that were used:")
            print(loaded_filenames)
            print("Time(s) that were used:")
            print(loaded_datetimes)

        start_time = time.time()
        loaded_datasets = self._reproject_datasets(loaded_datasets)
        if self.dataset_empty_or_out_of_range:
            return
        print("Reproj dataset time: " + str(time.time() - start_time))

        # Had to sort these into their respective label and example lists after the other loop becuase _reproject_datasets
        # needed to have all loaded datasets at the same time to function correctly
        for i, data_settings_cfg in enumerate(data_settings_cfgs):
            if data_settings_cfg["Data"]["is_label_data"]:
                loaded_datasets_labels.append(loaded_datasets[i])
            else:
                loaded_datasets_examples.append(loaded_datasets[i])
            loaded_datasets[i].close()
        loaded_datasets = None

        start_time = time.time()
        self.master_xarray_dataset_examples = self._create_master_xarray_dataset(loaded_datasets_examples, loaded_datetimes_examples, dataset_configs_examples)
        if self.master_xarray_dataset_incompatibility:
            return
        self.master_xarray_dataset_labels = self._create_master_xarray_dataset(loaded_datasets_labels, loaded_datetimes_labels, dataset_configs_labels)
        if self.master_xarray_dataset_incompatibility:
            return
        print("Master dataset time: " + str(time.time() - start_time))

        # Free up memory
        for ds in loaded_datasets_examples:
            ds.close()
        for ds in loaded_datasets_labels:
            ds.close()
        loaded_datasets_examples = None
        loaded_datasets_labels = None


    # Create an xarray dataset object that encompasses all of the currently loaded data from disk. 
    # This is to exploit the power of xarray datasets througout the rest of the patcher.
    def _create_master_xarray_dataset(self, loaded_datasets, loaded_datetimes, loaded_dataset_settings):
        self.master_xarray_dataset_incompatibility = False

        # Make the datetimes list only contain numpy datetimes and not my datetimepair objects for later
        loaded_datetimes_np = []
        for loaded_datetime in loaded_datetimes:
            loaded_datetimes_np.append(loaded_datetime.datetime_pair[0])
        loaded_datetimes = loaded_datetimes_np

        # Reorder our datasets into nested lists that contain data from the same 
        # dataset over its entire time period. So "loaded_datasets_ordered_by_time" can be descibed as a list of time series lists
        loaded_datasets_ordered_by_time = []
        loaded_datetimes_ordered_by_time = []
        dataset_names = []
        last_ds_name = ""
        single_ds_group_of_times = []
        single_ds_datetimes = []
        for i, data_settings_cfg in enumerate(loaded_dataset_settings):
            if data_settings_cfg["dataset_name"] != last_ds_name:
                if last_ds_name != "":
                    loaded_datasets_ordered_by_time.append(single_ds_group_of_times)
                    loaded_datetimes_ordered_by_time.append(np.array(single_ds_datetimes, dtype=np.datetime64))
                    for ds in single_ds_group_of_times:
                        ds.close()
                    single_ds_group_of_times = []
                    single_ds_datetimes = []
                dataset_names.append(data_settings_cfg["dataset_name"])
            single_ds_group_of_times.append(loaded_datasets[i])
            single_ds_datetimes.append(loaded_datetimes[i])
            last_ds_name = data_settings_cfg["dataset_name"]
            loaded_datasets[i].close()
        loaded_datasets_ordered_by_time.append(single_ds_group_of_times)
        loaded_datetimes_ordered_by_time.append(np.array(single_ds_datetimes, dtype=np.datetime64))

        # Create a list that keeps track of all lengths of time we are working with based on the number of xarray datasets in each of
        # the above described nested lists
        grouping_lens = []
        for loaded_dataset_group in loaded_datasets_ordered_by_time:
            grouping_lens.append(len(loaded_dataset_group))

        # Try to free up some memory
        for ds in single_ds_group_of_times:
            ds.close()
        for ds in loaded_datasets:
            ds.close()
        loaded_datasets = None
        single_ds_group_of_times = None

        dataset_list_to_merge = []
        if self.top_settings_patches["make_time_3d"]:
            # Check to make sure that we only have either datasets with one timestep or datasets with the same n number of timesteps
            unqiue_grouping_lens = np.unique(grouping_lens)
            if len(unqiue_grouping_lens) > 2 or (len(unqiue_grouping_lens) == 2 and 1 not in unqiue_grouping_lens):
                raise Exception("The time_offsets you have selected are not acceptable for the 3D patch case. You must have the same number of time_offsets in each dataset you want offsetted or a dataset should not have any offset.")

            for dataset_list in loaded_datasets_ordered_by_time:
                if len(dataset_list) == 1 and unqiue_grouping_lens[-1] != 1:
                    dataset_list_to_concat = [copy.deepcopy(dataset_list[0]) for i in range(unqiue_grouping_lens[-1])]
                    dataset_list_to_merge.append(xr.concat(dataset_list_to_concat, dim="time_dim"))
                elif len(unqiue_grouping_lens) == 1 and len(dataset_list) == 1:
                    dataset_list_to_merge.append(dataset_list[0])
                else:
                    dataset_list_to_merge.append(xr.concat(dataset_list, dim="time_dim"))

            xarray_dataset_datetimes = {"time": ("time_dim", loaded_datetimes_ordered_by_time[np.argmax(grouping_lens)])}

        else:
            for dataset_list in loaded_datasets_ordered_by_time:
                if len(dataset_list) == 1:
                    dataset_list_to_merge.append(dataset_list[0])
                else:
                    for i, dataset in enumerate(dataset_list):
                        new_ds_var_names = np.char.add(np.array(list(dataset.keys())), "_" + str(i))
                        dataset = dataset.rename_vars(dict(zip(list(dataset.keys()), new_ds_var_names)))
                        dataset_list_to_merge.append(dataset)
            
            xarray_dataset_datetimes = {}
            for i, dataset_name in enumerate(dataset_names):
                xarray_dataset_datetimes[dataset_name + "_time"] = (dataset_name + "_time_dim", loaded_datetimes_ordered_by_time[i])

        if len(dataset_list_to_merge) != 1:
            try:
                master_xarray_dataset = xr.merge(dataset_list_to_merge)
            except:
                for i, dataset_name in enumerate(dataset_names):
                    new_ds_var_names = np.char.add(np.array(list(dataset_list_to_merge[i].keys())), "_" + dataset_name)
                    dataset_list_to_merge[i] = dataset_list_to_merge[i].rename_vars(dict(zip(list(dataset_list_to_merge[i].keys()), new_ds_var_names)))
                try:
                    # Second try is for case where latlons actually don't line up and the search step has to be skipped.
                    master_xarray_dataset = xr.merge(dataset_list_to_merge)
                except:
                    self.master_xarray_dataset_incompatibility = True
                    return
        else:
            master_xarray_dataset = copy.deepcopy(dataset_list_to_merge[0])
        master_xarray_dataset = master_xarray_dataset.assign(xarray_dataset_datetimes)

        # Try to free up some memory
        for dataset_list in loaded_datasets_ordered_by_time:
            for ds in dataset_list:
                ds.close()
        loaded_datasets_ordered_by_time = None

        return master_xarray_dataset


    def _reproject_datasets(self, loaded_datasets):# , reproj_ds_index, data_settings_cfgs):
        reproj_datasets = []
        self.dataset_empty_or_out_of_range = False
        data_settings_cfgs = self.data_settings

        reproj_ds_index = -1
        for i, data_settings_cfg in enumerate(data_settings_cfgs):
            if data_settings_cfg["Data"]["reproj_target"]:
                reproj_ds_index = i
        if reproj_ds_index == -1:
            raise Exception("No dataset has been designated as the reproj_target. You must designate exactly one as this. Even if only loading one dataset.")

        reproj_target_ds = loaded_datasets[reproj_ds_index]

        for i, ds in enumerate(loaded_datasets):
            if data_settings_cfgs[i]["Data"]["needs_reproj"]:
                if data_settings_cfgs[i]["Data"]["reproj_target"]:
                    raise Exception('Cannot set both "needs_reproj" and "reproj_target" to True at the same time. You cannot reproject a dataset onto itself!')

                # TODO: Maybe make the regridder algorithm a config setting?
                # TODO: Consider if reuse_weights:bool is useful here
                regridder = xe.Regridder(ds, reproj_target_ds, "bilinear", unmapped_to_nan=True)
                ds_reproj = regridder(ds)
                for key_name in ds_reproj.keys():
                    data_var = ds_reproj[key_name].to_numpy()
                    if np.isnan(data_var).all():
                        self.dataset_empty_or_out_of_range = True
                        break

                if self.dataset_empty_or_out_of_range:
                    break
            else:
                ds_reproj = ds

            ds_reproj = self._add_custom_vars(ds_reproj, data_settings_cfgs, i, True)

            reproj_datasets.append(ds_reproj)

            if data_settings_cfgs[i]["Data"]["needs_reproj"]:
                ds.close()

        return reproj_datasets

    
    def get_valid_pixels(self, examples_ds, labels_ds):
        patch_size = self.top_settings_patches["patch_size"]
        for key in examples_ds.keys():
            if "time" in key:
                examples_ds = examples_ds.drop(key)
        for key in labels_ds.keys():
            if "time" in key:
                labels_ds = labels_ds.drop(key)
        for coord in examples_ds.coords:
            if "time" in coord:
                examples_ds = examples_ds.drop(coord)
        for coord in labels_ds.coords:
            if "time" in coord:
                labels_ds = labels_ds.drop(coord)

        try:
            ds = xr.merge([examples_ds, labels_ds])
        except:
            new_example_var_names = np.char.add(np.array(list(examples_ds.keys())), "_example")
            new_label_var_names = np.char.add(np.array(list(labels_ds.keys())), "_label")
            examples_ds = examples_ds.rename_vars(dict(zip(list(examples_ds.keys()), new_example_var_names)))
            labels_ds = labels_ds.rename_vars(dict(zip(list(labels_ds.keys()), new_label_var_names)))
            ds = xr.merge([examples_ds, labels_ds])

        lat_len = ds.dims["lat_dim"]
        lon_len = ds.dims["lon_dim"]

        if self.top_settings_patches["overlap_patches"]:
            valid_pixels = setup_valid_pixel_array(ds, self.top_settings_patches["maximized_dims"], True)
            valid_pixels[lon_len - patch_size:, :, ...] = 0
            valid_pixels[:, lat_len - patch_size:, ...] = 0
        else:
            valid_pixels = setup_valid_pixel_array(ds, self.top_settings_patches["maximized_dims"], False)
            x_indices = np.arange(0, lon_len - patch_size, patch_size)
            y_indices = np.arange(0, lat_len - patch_size, patch_size)
            for x in x_indices:
                for y in y_indices:
                    valid_pixels[x, y, ...] = 1

        if not self.top_settings_patches["ignore_nans"]:
            da = ds.reset_coords().to_array() #reset_coords is used because it forces the lat lons to also be included in nan checker.
            da = da.transpose("lat_dim", ...)
            da = da.transpose("lon_dim", ...)

            ds_dims_to_collapse = get_indices_of_dims_to_collapse(da, self.top_settings_patches["maximized_dims"])

            da = da.to_numpy()

            nan_mask = np.any(np.isnan(da), axis=tuple(ds_dims_to_collapse))
            kernelizer = Kernelizer(np.any, patch_size)
            nan_mask = kernelizer(nan_mask)
            valid_pixels = np.logical_and(valid_pixels, np.logical_not(nan_mask))
        
        for filter_str in self.top_settings_patches["filters"]:
            filter_import = importlib.import_module("filters." + filter_str)
            filter_mask = eval("filter_import." + filter_str + "(ds,self.top_settings_patches['maximized_dims'],patch_size)")

            if len(valid_pixels.shape) > 2 and len(filter_mask.shape) == 2:
                for i in np.arange(2,len(valid_pixels.shape)):
                    filter_mask = np.expand_dims(filter_mask, -1)
                filter_mask = np.tile(filter_mask, valid_pixels.shape[2:])
            
            if not filter_import.THIS_FILTER_RETURNS_PATCH_MASK:
                filter_mask = np.logical_not(filter_mask)
                kernelizer = Kernelizer(np.any, patch_size)
                filter_mask = kernelizer(filter_mask)
                filter_mask = np.logical_not(filter_mask)

            valid_pixels = np.logical_and(valid_pixels, filter_mask)
        
        valid_pixels_balanced = []
        for filter_str in self.top_settings_patches["filters_balanced"]:
            filter_import = importlib.import_module("filters." + filter_str)
            filter_mask = eval("filter_import." + filter_str + "(ds,self.top_settings_patches['maximized_dims'],patch_size)")

            if len(valid_pixels.shape) > 2 and len(filter_mask.shape) == 2:
                for i in np.arange(2,len(valid_pixels.shape)):
                    filter_mask = np.expand_dims(filter_mask, -1)
                filter_mask = np.tile(filter_mask, valid_pixels.shape[2:])
            
            if not filter_import.THIS_FILTER_RETURNS_PATCH_MASK:
                filter_mask = np.logical_not(filter_mask)
                kernelizer = Kernelizer(np.any, patch_size)
                filter_mask = kernelizer(filter_mask)
                filter_mask = np.logical_not(filter_mask)

            valid_pixels_balanced.append(np.logical_and(valid_pixels, filter_mask))
        
        if len(valid_pixels_balanced) == 0:
            valid_pixels_balanced = [valid_pixels]

        filtered_balanced_pixels = []
        for balanced_pixels in valid_pixels_balanced:
            where_array = np.array(np.where(balanced_pixels == 1))
            # TODO: Check this np.size if statement with some tests because I don't fully trust it
            if np.size(where_array):
                if self.top_settings_patches["shuffle_patches_in_each_timestep"]:
                    combined_lists_for_shuffle = list(zip(*where_array))
                    random.shuffle(combined_lists_for_shuffle)
                    array_of_where_coords = np.array(list(zip(*combined_lists_for_shuffle)))
                    filtered_balanced_pixels.append(array_of_where_coords)
                else:
                    filtered_balanced_pixels.append(where_array)
            else:
                filtered_balanced_pixels.append(np.array([[], []]))
        
        return filtered_balanced_pixels

    
    def _make_patches(self):
        filtered_balanced_counts = self.filtered_balanced_counts
        master_examples_dataset = self.master_xarray_dataset_examples
        master_labels_dataset = self.master_xarray_dataset_labels
        patch_size = self.top_settings_patches["patch_size"]

        pixel_counters = np.zeros(len(filtered_balanced_counts), dtype=np.int64)

        # NOTE: It seems that filtered_balanced_pixels is required to be a list of len "number of filters" that each contain a list of size 2
        # (for x and y coords respectively) that each contain a list of all the found pixels (of len "number of found pixels")
        start_time = time.time()
        filtered_balanced_pixels = self.get_valid_pixels(master_examples_dataset, master_labels_dataset)
        print("Find valid patches time: " + str(time.time() - start_time))
        
        start_time = time.time()
        example_patches = []
        label_patches = []
        for i in range(self.patches_per_time):
            filter_balance_order = np.array(filtered_balanced_counts).argsort()
            for filter_balance_ind in filter_balance_order:
                if filtered_balanced_counts[filter_balance_ind] < self.number_of_patches_per_balanced_var and len(filtered_balanced_pixels[filter_balance_ind][0]) > 0 and pixel_counters[filter_balance_ind] < len(filtered_balanced_pixels[filter_balance_ind][0]):
                    x_i = filtered_balanced_pixels[filter_balance_ind][0][pixel_counters[filter_balance_ind]]
                    y_i = filtered_balanced_pixels[filter_balance_ind][1][pixel_counters[filter_balance_ind]]
           
                    example_patch = self._make_patch(master_examples_dataset, patch_size, x_i, y_i)
                    label_patch = self._make_patch(master_labels_dataset, patch_size, x_i, y_i)

                    if filtered_balanced_pixels[filter_balance_ind].shape[0] > 2:
                        for non_latlon_dim_index in np.arange(2, filtered_balanced_pixels[filter_balance_ind].shape[0]):
                            if self.top_settings_patches["maximized_dims"][non_latlon_dim_index-2] in example_patch.dims:
                                example_patch = example_patch[{self.top_settings_patches["maximized_dims"][non_latlon_dim_index-2]: filtered_balanced_pixels[filter_balance_ind][non_latlon_dim_index][pixel_counters[filter_balance_ind]]}]
                            if self.top_settings_patches["maximized_dims"][non_latlon_dim_index-2] in label_patch.dims:
                                label_patch = label_patch[{self.top_settings_patches["maximized_dims"][non_latlon_dim_index-2]: filtered_balanced_pixels[filter_balance_ind][non_latlon_dim_index][pixel_counters[filter_balance_ind]]}]

                    example_patches.append(example_patch)
                    label_patches.append(label_patch)

                    filtered_balanced_counts[filter_balance_ind] = filtered_balanced_counts[filter_balance_ind] + 1
                    pixel_counters[filter_balance_ind] = pixel_counters[filter_balance_ind] + 1
                    break

        if np.sum(pixel_counters) < self.patches_per_time:
            warnings.warn("While generating patches for a single timestep, the function _make_patches ran out of possible patches that meet the set filters' requirements. Continuing search...")

        print("Accepted Patches: " + str(pixel_counters))

        self.feature_patches = self._concat_patches(self.feature_patches, example_patches)
        self.label_patches = self._concat_patches(self.label_patches, label_patches)

        print("Make patches time: " + str(time.time() - start_time))

        # Free up memory
        for patch in example_patches:
            patch.close()
        for patch in label_patches:
            patch.close()
        master_examples_dataset.close()
        master_examples_dataset = None
        master_labels_dataset.close()
        master_labels_dataset = None
        self.master_xarray_dataset_examples.close()
        self.master_xarray_dataset_labels.close()
        self.master_xarray_dataset_examples = None
        self.master_xarray_dataset_labels = None
        self.filtered_balanced_counts = filtered_balanced_counts

    
    def _concat_patches(self, patches, patch):
        if patch is None or (type(patch) is list and len(patch) == 0):
            return patches
        if patches is None:
            if type(patch) is xr.Dataset:
                patches = copy.deepcopy(patch.expand_dims(dim='n_samples'))
                patch.close()
            else:
                patches = xr.concat(patch, dim='n_samples')
        else:
            if type(patch) is xr.Dataset:
                patches = xr.concat([patches, patch.expand_dims(dim='n_samples')], dim='n_samples')
            else:
                patches = [patches]
                patches.extend(patch)
                patches = xr.concat(patches, dim='n_samples')

        return patches
    

    def _make_patch(self, file_ds, patch_size, x, y):
        patch = file_ds[{"lon_dim":slice(x, x+patch_size), "lat_dim":slice(y, y+patch_size)}]

        return patch


    # Get the list of all files paths and create all DatetimePair objects for all scenarios (including when datetimes must be loaded from disk).
    def get_files_and_datetimes(self):
        self.datasets_paths = []
        self.datasets_datetimes = []
        self.dataset_pos_in_file = []
        self.dataset_full_list_pos = []

        for i, data_settings_cfg in enumerate(self.data_settings):
            file_list = self._create_file_list(data_settings_cfg["Path"]["root_dir"], 
                                                data_settings_cfg["Path"]["path_glob"], 
                                                data_settings_cfg["Path"]["path_reg"])

            init_dateset_datetimes = None
            init_dataset_date_resolution = None
            init_dataset_date_resolution_val = None

            # This block is for loading initialization times if they are present
            if data_settings_cfg["Path"]["init_dt_positions"] is not None and data_settings_cfg["Path"]["init_dt_regs"] is not None and data_settings_cfg["Path"]["init_dt_formats"] is not None:
                init_dateset_datetimes, init_dataset_date_resolution, init_dataset_date_resolution_val = self._extract_best_datetime_no_IO(data_settings_cfg["Path"]["root_dir"],
                                                                                                                        file_list, data_settings_cfg["Path"]["init_dt_positions"],
                                                                                                                        data_settings_cfg["Path"]["init_dt_regs"],
                                                                                                                        data_settings_cfg["Path"]["init_dt_formats"])

            # The if condition below is for the case where the patcher to load datetime data from within the dataset netcdf files themselves
            if data_settings_cfg["Data"]["has_time_cord"]:

                # Make the complete path to the datetime netcdf using the given datetime_netcdf_dir and the dataset name
                datetime_netcdf_dir = self.top_settings_input["datetime_netcdf_dir"]
                datetime_netcdf_path = os.path.join(datetime_netcdf_dir, self.top_settings_input["dataset_names"][i] + ".nc")

                # Pull the datetime information from already created datetime netcdf
                if os.path.exists(datetime_netcdf_path):
                    dateset_datetimes, ds_full_pos, ds_pos_in_file = self.load_datetimes_from_pre_saved_netcdf(datetime_netcdf_path, data_settings_cfg["Data"]["time_cord_name"])
                    dataset_date_resolution, dataset_date_resolution_val = self._find_datetime_resolution(data_settings_cfg["Path"]["internal_dt_positions"])
                # Make new datetime netcdf since none is present. NOTE: THIS IS A LONG PROCESS THAT CANNOT BE DONE IN PARALLEL. THIS KILLS THE PROCESS WHEN DONE
                else:
                    self.make_datetime_netcdf_file(file_list, 0, data_settings_cfg["Data"]["y_dim_name"], data_settings_cfg["Data"]["x_dim_name"],
                                                   data_settings_cfg["Data"]["time_cord_name"], data_settings_cfg["Data"]["time_dim_name"], datetime_netcdf_path)

            # The else condition is for when time only has to be pulled from the path name strings themselves
            else:
                dateset_datetimes, dataset_date_resolution, dataset_date_resolution_val = self._extract_best_datetime_no_IO(data_settings_cfg["Path"]["root_dir"],
                                                                                                                            file_list, data_settings_cfg["Path"]["dt_positions"],
                                                                                                                            data_settings_cfg["Path"]["dt_regs"],
                                                                                                                            data_settings_cfg["Path"]["dt_formats"])
                ds_full_pos = np.arange(len(dateset_datetimes))
                ds_pos_in_file = np.zeros(len(dateset_datetimes), dtype=np.int64)



            dateset_datetimes, init_dateset_datetimes, [ds_full_pos, ds_pos_in_file] = self._select_data_range(self.dataset_start_times[i], self.dataset_end_times[i],
                                                                                                dateset_datetimes, data_settings_cfg["Bounds"]["use_date_for_data_range"], 
                                                                                                dataset_date_resolution, init_dateset_datetimes, [ds_full_pos, ds_pos_in_file])


            if len(dateset_datetimes) == 0:
                raise Exception('No files found under given "data_start" and "data_end" settings for at least one dataset, despite files being found given your glob and regex settings. If using dates, maybe they are incorrect?')
            
            # This block creates the actual DatetimePair objects
            datetime_pair_obj_list = []
            for j, dateset_datetime in enumerate(dateset_datetimes):
                if init_dateset_datetimes is not None:
                    init_dateset_datetime = init_dateset_datetimes[j]
                else:
                    init_dateset_datetime = None
                datetime_pair_obj = DatetimePair(dateset_datetime, dataset_date_resolution, dataset_date_resolution_val, 
                                    init_dateset_datetime, init_dataset_date_resolution, init_dataset_date_resolution_val)
                datetime_pair_obj_list.append(datetime_pair_obj)

            # Assign everything to class fields
            self.datasets_paths.append(file_list)
            self.datasets_datetimes.append(np.array(datetime_pair_obj_list))
            self.dataset_pos_in_file.append(ds_pos_in_file)
            self.dataset_full_list_pos.append(ds_full_pos)


    # Three possible options:
    # 1. use dates to set range
    # 2. use hard indeces to set range
    # 3. use percentages (real numbers) to set range
    # NOTE: Does not check validity of selections. May add that later
    # NOTE: Please make sure the start and end dates in ISO_8601 format
    # Other_lists is a list of (right now numpy arrays, see NOTE below) that contain non-datetime objects that also need to be cut by the date range
    def _select_data_range(self, data_start, data_end, dates_list, use_date=False, dataset_date_resolution=None, init_date_list=None, other_lists=None):
        # NOTE: Other lists assumes to be numpy arrays. If I change them to be actual python lists becuase of the 
        # legacy reasons discussed in the netcdf datetime file method, then an array cast will be needed below.
        new_other_lists = []

        # If date is used for selecting files, this assumes that the dates are included and are not None
        if use_date:
            if data_start is not None:
                start_date = np.datetime64(data_start).astype(dataset_date_resolution)
            if data_end is not None:
                end_date = np.datetime64(data_end).astype(dataset_date_resolution)

            if data_start is not None and data_end is not None:
                inds = np.where(np.logical_and(dates_list >= start_date, dates_list <= end_date))
                if init_date_list is not None:
                    init_date_list = init_date_list[inds]
                if other_lists is not None:
                    for other_list in other_lists:
                        new_other_lists.append(other_list[inds])
                    other_lists = new_other_lists
                return dates_list[inds], init_date_list, other_lists
            
            elif data_start is None and data_end is not None:
                inds = np.where(dates_list <= end_date)
                if init_date_list is not None:
                    init_date_list = init_date_list[inds]
                if other_lists is not None:
                    for other_list in other_lists:
                        new_other_lists.append(other_list[inds])
                    other_lists = new_other_lists
                return dates_list[inds], init_date_list, other_lists
            
            elif data_start is not None and data_end is None:
                inds = np.where(dates_list >= start_date)
                if init_date_list is not None:
                    init_date_list = init_date_list[inds]
                if other_lists is not None:
                    for other_list in other_lists:
                        new_other_lists.append(other_list[inds])
                    other_lists = new_other_lists
                return dates_list[inds], init_date_list, other_lists
            
            else:
                return dates_list, init_date_list, other_lists
        
        else:
            start_index = 0
            end_index = -1

            if data_start is not None:
                if isinstance(data_start, int):
                    start_index = data_start
                elif isinstance(data_start, float):
                    start_index = int(data_start*len(dates_list))

            if data_end is not None:
                if isinstance(data_end, int):
                    end_index = data_end
                elif isinstance(data_end, float):
                    end_index = int(data_end*len(dates_list))
        
            if dates_list is not None:
                dates_list = dates_list[start_index:end_index]
            
            if init_date_list is not None:
                init_date_list = init_date_list[start_index:end_index]
            
            if other_lists is not None:
                for other_list in other_lists:
                    new_other_lists.append(other_list[start_index:end_index])
                other_lists = new_other_lists

            return dates_list, init_date_list, other_lists

    
    def _find_datetime_resolution(self, datetime_char):
        # TODO: Double check datetime64[char] have been chosen right
        if "m" in datetime_char:
            dataset_date_resolution = "datetime64[m]"
            dataset_date_resolution_val = 4
        elif "h" in datetime_char:
            dataset_date_resolution = "datetime64[h]"
            dataset_date_resolution_val = 3
        elif "D" in datetime_char:
            dataset_date_resolution = "datetime64[D]"
            dataset_date_resolution_val = 2
        elif "M" in datetime_char:
            dataset_date_resolution = "datetime64[M]"
            dataset_date_resolution_val = 1
        else:
            dataset_date_resolution = "datetime64[Y]"
            dataset_date_resolution_val = 0

        return dataset_date_resolution, dataset_date_resolution_val


    # Extracts the best possible time information from data's directory structure and file name 
    # NOTE: See the required characters for designating where each datetime component is located in each file's path
    # NOTE: If a lower level datetime unit is used (for example hours or minute), every higher level must also be present
    # NOTE: datetime_positions chars differ from the datetime.datetime chars needed for "datetime_ISO_formats". May change this later
    # NOTE: In the path/name of each file there must be at least SOME datetime information. The no-information scenario is not currently supported
    def _extract_best_datetime_no_IO(self, root_path, data_file_list, datetime_positions, extraction_regs, datetime_ISO_formats):
        datetime_chars = datetime_positions.split("/")
        root_len = len(root_path.rstrip("/").split("/"))

        datetime_chars_indeces = []
        datetime_chars_seperated = []
        for i, datetime_str in enumerate(datetime_chars):
            for datetime_char in datetime_str:
                if datetime_char not in ["Y","M","D","h","m"]:
                    raise Exception("A given datetime position character is not one of the accepted options.")
                datetime_chars_seperated.append(datetime_char)
                datetime_chars_indeces.append(i + root_len)

        if len(datetime_chars_indeces) == 0:
            raise Exception("Giving no datetime information in the file names or in each file's path is not supported.")

        dataset_date_resolution, dataset_date_resolution_val = self._find_datetime_resolution(datetime_chars_seperated)

        files_to_remove = []
        datetimes = []
        for data_file in data_file_list:
            file_datetime_read_fail = False
            data_file_split = data_file.split("/")

            minute_str = ""
            hour_str = ""
            day_str = ""
            month_str = ""
            year_str = ""
            
            for i, datetime_char in enumerate(datetime_chars_seperated):
                reg_extracted = re.search(extraction_regs[i], data_file_split[datetime_chars_indeces[i]])
                if not reg_extracted:
                    file_datetime_read_fail = True
                    break
                time_component_str = reg_extracted.group(1)

                if datetime_char == "Y":
                    datetime_obj = datetime.strptime(time_component_str, datetime_ISO_formats[i])
                    year_str = datetime_obj.strftime("%Y")
                elif datetime_char == "M":
                    datetime_obj = datetime.strptime(time_component_str, datetime_ISO_formats[i])
                    month_str = "-" + datetime_obj.strftime("%m")
                elif datetime_char == "D":
                    datetime_obj = datetime.strptime(time_component_str, datetime_ISO_formats[i])
                    day_str = "-" + datetime_obj.strftime("%d")
                elif datetime_char == "h":
                    datetime_obj = datetime.strptime(time_component_str, datetime_ISO_formats[i])
                    hour_str = "T" + datetime_obj.strftime("%H")
                elif datetime_char == "m":
                    datetime_obj = datetime.strptime(time_component_str, datetime_ISO_formats[i])
                    minute_str = ":" + datetime_obj.strftime("%M")

            if file_datetime_read_fail:
                files_to_remove.append(data_file)
                continue

            datetime_np_str = year_str + month_str + day_str + hour_str + minute_str

            try:
                datetime_np = np.datetime64(datetime_np_str)
            except:
                files_to_remove.append(data_file)
                continue

            datetimes.append(datetime_np)

        if len(datetimes) == 0:
            raise Exception('Failed to extract datetime values using "dt_positions", "dt_regs", and "dt_formats" for at least one dataset. Probably have a mistake in your regular expressions.')

        return np.array(datetimes), dataset_date_resolution, dataset_date_resolution_val


    # NOTE: In path_glob only include wild card operators for each directory level you want to search across.
    #       The regex can handle any filtering.
    def _create_file_list(self, root_dir, path_glob, path_regex):
        glob_path = root_dir.rstrip("/") + "/" + path_glob.lstrip("/")
        unfiltered_file_list = glob.glob(glob_path)

        file_list = []
        wildcard_indeces = np.where(np.array(glob_path.split("/")) == "*")[0]
        for file in unfiltered_file_list:
            file_array = np.array(file.split("/"))
            reg_bools = []
            for i, reg in enumerate(path_regex):
                reg_bool = re.search(reg, file_array[wildcard_indeces[i]])
                if reg_bool:
                    reg_bools.append(True)
                else:
                    reg_bools.append(False)

            if np.all(np.array(reg_bools)):
                file_list.append(file)

        if len(file_list) == 0:
            raise Exception('File loading issue with at least one dataset! Either the wildcards in "path_glob" are in the wrong positions in one of the dataset configs or perhaps the disk you are trying to access is no longer mounted!')
        
        file_list.sort()
        return file_list


'''
Next two functions functions I wrote while at UBC. Useful but maybe could be updated.
'''

# NOTE: This method assumes that the config will never have things nested
# deeper than one level. This should always be the case in the .cfg anyways
def cfg_parser(cfg_object):
    new_cfg = {}
    cfg = cfg_object._sections
    for key_i, value_i in cfg.items():
        if type(value_i) is dict:
            for key_j, value_j in value_i.items():
                if key_i not in new_cfg:
                    new_cfg[key_i] = {}
                new_cfg[key_i][key_j] = value_parser(value_j)
        else:
            new_cfg[key_i] = value_parser(value_i)
    
    return new_cfg


# Not my favorite solution, but it is easy to read and understand.
def value_parser(value):
    try:
        return int(value)
    except:
        pass
    try:
        return float(value)
    except:
        pass
    if value.lower() == "false":
        return False
    if value.lower() == "true":
        return True
    if value.lower() == "none":
        return None
    if value[0] == '"' and value[-1] == '"' and len(value) >= 2:
        value = re.sub(".$", "", value)
        value = re.sub("^.", "", value)
        return value
    elif value[0] == "'" and value[-1] == "'" and len(value) >= 2:
        value = re.sub(".$", "", value)
        value = re.sub("^.", "", value)
        return value
    elif value[0] == "[" and value[-1] == "]":
        if len(value) == 2:
            return []
        else:
            # TODO: I am not pleased with this solution at all but it will do until I can be bothered to make it more secure.
            return eval(value)
    raise Exception("value_parser failed. Invalid config setting: " + value)


if __name__ == "__main__":
         # Parse the command-line arguments
    parser = argparse.ArgumentParser(description='NetCDF Patch Generator')
    parser.add_argument('--run_num', type=int, help='Number to label this run')
    parser.add_argument('--config_path', type=str, help='Path to config file')
    args = parser.parse_args()

    # TODO: Switch this to command line argument
    patcher = Patcher(args.run_num, args.config_path)
    patcher.run()