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
from operator import itemgetter


# NOTE: netcdf4 also need to be manually installed as dependencies
#       IMPORTANT: First thing to install on new env is xesmf/esmpy using the command conda install -c conda-forge xesmf esmpy=8.0.0
#       this will install python as well. Second thing to install is cftime. Third thing is to install dask. Then see netcdf4 above.  
#       tensorflow must also be version 2.7.0 or greater (if using keras-unet-collection in another script).

# NOTE: Spatial dataset delineation in file/path naming not currently supported.


# TODO: Make documentation on how data directories should be organized. (Data downloader should make them this way)
#       They should be made such that the full date (but not necessarily the hour) is always either a directory 
#       (or a set of directories) somewhere in each file's path and/or be included in the name of each file.
#       It should note that it will fail if those .idx files are in the label/feature directories. Only expected files should be in there.
class Patcher:
    def __init__(self, run_num):
        # # Parse in config file specified by config_path. See examples given in repo
        # config = configparser.ConfigParser()
        # config.read(config_path)
        # self.config = cfg_parser(config)
        self.run_num = run_num
        # warnings.simplefilter('once', category=RuntimeWarning)


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


    # NOTE: Only works in multi-dateset case. Not to be used for 1 dataset case.
    def _compare_datetimes_with_IO(self, current_index, chosen_date_indeces, all_datetimes,
                                         data_settings_cfgs, datasets_paths, chosen_resolution, 
                                         solution_indeces_files=None, solution_indeces_times=None, 
                                         chosen_datetimes_adjusted=None, all_found_datetimes_adjusted = None, 
                                         all_found_time_indeces_adjusted = None):
        found_files = False
        if solution_indeces_files is None:
            solution_indeces_files = np.zeros(len(chosen_date_indeces), dtype=np.int64)
        if solution_indeces_times is None:
            solution_indeces_times = np.zeros(len(chosen_date_indeces), dtype=np.int64)
        if chosen_datetimes_adjusted is None:
            chosen_datetimes_adjusted = []
            for i in range(len(chosen_date_indeces)):
                chosen_datetimes_adjusted.append(None)
        # TODO: Double check that the setup done below is correct (AND THAT LOGIC IN LOOP WHERE THIS FUNCTION IS CALLED IS CORRECT)
        if all_found_datetimes_adjusted is None:
            all_found_datetimes_adjusted = []
            all_found_time_indeces_adjusted = []
            for i in range(len(datasets_paths)):
                all_found_datetimes_adjusted_cols = []
                all_found_time_indeces_adjusted_cols = []
                all_found_datetimes_adjusted.append(all_found_datetimes_adjusted_cols)
                all_found_time_indeces_adjusted.append(all_found_time_indeces_adjusted_cols)
                for j in range(len(datasets_paths[i])):
                    all_found_datetimes_adjusted[i].append(None)
                    all_found_time_indeces_adjusted[i].append(None)

        for i in range(solution_indeces_files[current_index], len(chosen_date_indeces[current_index])):
            if data_settings_cfgs[current_index]["Data"]["use_internal_times_when_finding_files"]:
                if all_found_datetimes_adjusted[current_index][chosen_date_indeces[current_index][i]] is None:
                    time_cord_name = data_settings_cfgs[current_index]["Data"]["time_cord_name"]
                    y_dim_name = data_settings_cfgs[current_index]["Data"]["y_dim_name"]
                    x_dim_name = data_settings_cfgs[current_index]["Data"]["x_dim_name"]
                    ds = self._netcdf_loader(datasets_paths[current_index][chosen_date_indeces[current_index][i]], current_index, y_dim_name, x_dim_name)
                    # TODO: This is perhaps an extra loop. Consider if this can be removed later.
                    datetimes_adjusted = []
                    time_indeces_adjusted = np.arange(len(ds[time_cord_name].to_numpy())).tolist()
                    for current_datetime in ds[time_cord_name].to_numpy():
                        datetimes_adjusted.append(current_datetime.astype(chosen_resolution))
                    combined_lists_for_shuffle = list(zip(datetimes_adjusted, time_indeces_adjusted))
                    random.shuffle(combined_lists_for_shuffle)
                    datetimes_adjusted, time_indeces_adjusted = zip(*combined_lists_for_shuffle)
                    datetimes_adjusted, time_indeces_adjusted = list(datetimes_adjusted), list(time_indeces_adjusted)

                    all_found_datetimes_adjusted[current_index][chosen_date_indeces[current_index][i]] = datetimes_adjusted
                    all_found_time_indeces_adjusted[current_index][chosen_date_indeces[current_index][i]] = time_indeces_adjusted
                    ds.close()

                for j in range(solution_indeces_times[current_index], len(all_found_datetimes_adjusted[current_index][chosen_date_indeces[current_index][i]])):
                    current_datetime = all_found_datetimes_adjusted[current_index][chosen_date_indeces[current_index][i]][j]
                    chosen_datetimes_adjusted[current_index] = current_datetime

                    if current_index == 0:
                        found_files, solution_indeces_files, solution_indeces_times, all_found_datetimes_adjusted, all_found_time_indeces_adjusted = self._compare_datetimes_with_IO(
                                                                                                                                                    current_index + 1, chosen_date_indeces, all_datetimes, 
                                                                                                                                                    data_settings_cfgs, datasets_paths, 
                                                                                                                                                    chosen_resolution, solution_indeces_files, 
                                                                                                                                                    solution_indeces_times, chosen_datetimes_adjusted, 
                                                                                                                                                    all_found_datetimes_adjusted, all_found_time_indeces_adjusted)

                    elif chosen_datetimes_adjusted[current_index-1] == current_datetime:
                        if current_index == len(chosen_date_indeces)-1:
                            found_files = True

                        else:
                            found_files, solution_indeces_files, solution_indeces_times, all_found_datetimes_adjusted, all_found_time_indeces_adjusted = self._compare_datetimes_with_IO(
                                                                                                                                                        current_index + 1, chosen_date_indeces, all_datetimes, 
                                                                                                                                                        data_settings_cfgs, datasets_paths, 
                                                                                                                                                        chosen_resolution, solution_indeces_files, 
                                                                                                                                                        solution_indeces_times, chosen_datetimes_adjusted, 
                                                                                                                                                        all_found_datetimes_adjusted, all_found_time_indeces_adjusted)

                    if found_files:
                        solution_indeces_times[current_index] = all_found_time_indeces_adjusted[current_index][chosen_date_indeces[current_index][i]][j]
                        break

            else:
                current_datetime = all_datetimes[current_index][chosen_date_indeces[current_index][i]].astype(chosen_resolution)
                chosen_datetimes_adjusted[current_index] = current_datetime

                if current_index == 0:
                    found_files, solution_indeces_files, solution_indeces_times, all_found_datetimes_adjusted, all_found_time_indeces_adjusted = self._compare_datetimes_with_IO(
                                                                                                                                                current_index + 1, chosen_date_indeces, all_datetimes, 
                                                                                                                                                data_settings_cfgs, datasets_paths, 
                                                                                                                                                chosen_resolution, solution_indeces_files, 
                                                                                                                                                solution_indeces_times, chosen_datetimes_adjusted, 
                                                                                                                                                all_found_datetimes_adjusted, all_found_time_indeces_adjusted)
                    if found_files and data_settings_cfgs[current_index]["Data"]["has_time_cord"]:
                        y_dim_name = data_settings_cfgs[current_index]["Data"]["y_dim_name"]
                        x_dim_name = data_settings_cfgs[current_index]["Data"]["x_dim_name"]
                        ds = self._netcdf_loader(datasets_paths[current_index][chosen_date_indeces[current_index][i]], current_index, y_dim_name, x_dim_name)
                        time_cord_name = data_settings_cfgs[current_index]["Data"]["time_cord_name"]
                        for j, current_datetime in enumerate(ds[time_cord_name].to_numpy()):
                            if current_datetime.astype(chosen_resolution) == chosen_datetimes_adjusted[current_index+1]:
                                solution_indeces_times[current_index] = j
                                break
                            if j == len(ds[time_cord_name].to_numpy())-1:
                                raise Exception('Selected "False" for "use_internal_times_when_finding_files" along with "True" for "has_time_cord" for one or more of the given datasets however at least one file with time in it was not temporally compatible with the rest. Perhaps it was mislabeled?')
                        ds.close()

                elif chosen_datetimes_adjusted[current_index-1] == current_datetime:
                    if current_index == len(chosen_date_indeces)-1:
                        found_files = True

                    else:
                        found_files, solution_indeces_files, solution_indeces_times, all_found_datetimes_adjusted, all_found_time_indeces_adjusted = self._compare_datetimes_with_IO(
                                                                                                                                                    current_index + 1, chosen_date_indeces, all_datetimes, 
                                                                                                                                                    data_settings_cfgs, datasets_paths, 
                                                                                                                                                    chosen_resolution, solution_indeces_files, 
                                                                                                                                                    solution_indeces_times, chosen_datetimes_adjusted, 
                                                                                                                                                    all_found_datetimes_adjusted, all_found_time_indeces_adjusted) 
                    if found_files and data_settings_cfgs[current_index]["Data"]["has_time_cord"]:
                        y_dim_name = data_settings_cfgs[current_index]["Data"]["y_dim_name"]
                        x_dim_name = data_settings_cfgs[current_index]["Data"]["x_dim_name"]
                        ds = self._netcdf_loader(datasets_paths[current_index][chosen_date_indeces[current_index][i]], current_index, y_dim_name, x_dim_name)
                        time_cord_name = data_settings_cfgs[current_index]["Data"]["time_cord_name"]
                        for j, current_datetime in enumerate(ds[time_cord_name].to_numpy()):
                            if current_datetime.astype(chosen_resolution) == chosen_datetimes_adjusted[current_index-1]:
                                solution_indeces_times[current_index] = j
                                break
                            if j == len(ds[time_cord_name].to_numpy())-1:
                                raise Exception('Selected "False" for "use_internal_times_when_finding_files" along with "True" for "has_time_cord" for one or more of the given datasets however at least one file with time in it was not temporally compatible with the rest. Perhaps it was mislabeled?')                                     
                        ds.close()

            if found_files:
                solution_indeces_files[current_index] = i
                break
            else:
                solution_indeces_times[current_index] = 0

        if not found_files:
            solution_indeces_files[current_index] = 0


        return found_files, solution_indeces_files, solution_indeces_times, all_found_datetimes_adjusted, all_found_time_indeces_adjusted

    
    def run(self, settings_dict):
        patch_size = settings_dict["Patches"]["patch_size"]
        feature_patches_root = settings_dict["Output"]["examples_root"]
        label_patches_root = settings_dict["Output"]["labels_root"]
        n_patches = settings_dict["Patches"]["number_of_patches"]
        max_num_of_searches = settings_dict["Stopping"]["max_num_of_searches"]
        patches_per_time = settings_dict["Patches"]["patches_per_unit_time"]
        chosen_resolution = settings_dict["Patches"]["chosen_resolution"]
        self.dataset_names = settings_dict["Input_Data"]["dataset_names"]
        n_parallel_runs = settings_dict["Patches"]["n_parallel_runs"]
        self.shuffle_patches_in_each_timestep = settings_dict["Patches"]["shuffle_patches_in_each_timestep"]
        if settings_dict["Patches"]["max_times_num_per_file"] is None: # NOTE: This is for the number of specific times alowed to be extraced from one set of files. NOT NUMBER OF PATCHES ALLOWED PER FILE
            max_times_num_per_file = np.inf
        else:
            max_times_num_per_file = settings_dict["Patches"]["max_times_num_per_file"]

        data_settings_cfgs = []
        for data_settings_cfg in settings_dict["Input_Data"]["input_cfg_list"]:
            config = configparser.ConfigParser()
            config.read(data_settings_cfg)
            data_settings_cfgs.append(cfg_parser(config))

        self.dataset_netcdf_load_modes = np.zeros(len(data_settings_cfgs))

        # TODO: Maybe put the following loop into its own method?
        datasets_paths = []
        datasets_datetimes = []
        datasets_date_resolutions = []
        datasets_date_resolution_vals = []
        filtered_balanced_counts = []
        for data_settings_cfg in data_settings_cfgs:
            file_list = self.create_file_list(data_settings_cfg["Path"]["root_dir"], 
                                              data_settings_cfg["Path"]["path_glob"], 
                                              data_settings_cfg["Path"]["path_reg"])

            dateset_datetimes, dataset_date_resolution, dataset_date_resolution_val = self.extract_best_datetime_no_IO(data_settings_cfg["Path"]["root_dir"],
                                                                                                                       file_list, data_settings_cfg["Path"]["dt_positions"],
                                                                                                                       data_settings_cfg["Path"]["dt_regs"],
                                                                                                                       data_settings_cfg["Path"]["dt_formats"])

            file_list, dateset_datetimes = self.select_data_range(file_list, data_settings_cfg["Bounds"]["data_start"], data_settings_cfg["Bounds"]["data_end"],
                                                                  data_settings_cfg["Bounds"]["use_date_for_data_range"], dateset_datetimes, dataset_date_resolution)

            datasets_paths.append(file_list)
            datasets_datetimes.append(dateset_datetimes)
            datasets_date_resolutions.append(dataset_date_resolution)
            datasets_date_resolution_vals.append(dataset_date_resolution_val)
            for i in data_settings_cfg["Filtration"]["filters_balanced"]:
                filtered_balanced_counts.append(0)
        
        if len(filtered_balanced_counts) == 0:
            filtered_balanced_counts = [0]

        # Set top level lists to numpy arrays for more functionality
        datasets_paths = np.array(datasets_paths, dtype=list)
        datasets_datetimes = np.array(datasets_datetimes, dtype=list)
        datasets_date_resolutions = np.array(datasets_date_resolutions)
        datasets_date_resolution_vals = np.array(datasets_date_resolution_vals)
        data_settings_cfgs = np.array(data_settings_cfgs)

        # Sort by how high each dataset's resolution is
        inds = datasets_date_resolution_vals.argsort()
        datasets_paths = datasets_paths[inds]
        datasets_datetimes = datasets_datetimes[inds]
        datasets_date_resolutions = datasets_date_resolutions[inds]
        datasets_date_resolution_vals = datasets_date_resolution_vals[inds]
        data_settings_cfgs = data_settings_cfgs[inds]

        # If doing parallel runs on supercomputer, split our data across the runs so there is no double sampling of patches
        if n_parallel_runs is not None:
            datasets_paths_split = np.array_split(datasets_paths[-1], n_parallel_runs)
            datasets_datetimes_split = np.array_split(datasets_datetimes[-1], n_parallel_runs)
            datasets_paths[-1] = datasets_paths_split[self.run_num]
            datasets_datetimes[-1] = datasets_datetimes_split[self.run_num]

        # Have numpy datetime64 objects with dates adjusted for the lowest resolution we have ready for all datasets
        # TODO: Double check this idea by considering it once more
        lowest_resolution_dates = []
        for dataset_datetimes in datasets_datetimes:
            datetimes_adjusted = []
            for dataset_datetime in dataset_datetimes:
                datetimes_adjusted.append(dataset_datetime.astype(datasets_date_resolutions[0]))
            lowest_resolution_dates.append(datetimes_adjusted)

        all_found_datetimes_adjusted = None
        all_found_time_indeces_adjusted = None
        
        # Main high level variables that govern the flow of patching.
        load_new_files = True
        found_files = False
        solution_indeces_files = None
        solution_indeces_times = None
        date_indeces = np.random.choice(np.arange(0,len(datasets_paths[-1])), size=len(datasets_paths[-1]), replace=False)
        time_indeces = None
        chosen_date_indeces = []
        for i in range(len(datasets_paths)):
            chosen_date_indeces.append([])
        date_counter = 0
        data_per_file_counter = 0
        main_loop_counter = 0
        # Setup solution_indeces_files and solution_indeces_times for case where we only are loading one dataset
        if len(chosen_date_indeces) == 1:
            solution_indeces_files = date_indeces
        
        #TODO: Make a bunch of checks here that will throw full exceptions if stuff missing that is needed for the main loop.
        # For example check if the dataset(s) are all empty

        number_of_patches_per_balanced_var = n_patches / len(filtered_balanced_counts)
        feature_patches = None
        label_patches = None

        while np.any(np.array(filtered_balanced_counts) < number_of_patches_per_balanced_var):
            if main_loop_counter % 5 == 0:
                print("Reached search number: " + str(main_loop_counter))

            if main_loop_counter == max_num_of_searches:
                warnings.warn('WARNING: Hit maximum number of allowed searches set by "max_num_of_searches". Number of completed patches may be less than expected.')
                break
            main_loop_counter = main_loop_counter + 1
        
            counters_dict = {"date_counter": date_counter,
                             "data_per_file_counter": data_per_file_counter}
            flags_dict = {"load_new_files": load_new_files,
                          "found_files": found_files}
            index_dict = {"solution_indeces_files": solution_indeces_files,
                          "solution_indeces_times": solution_indeces_times,
                          "date_indeces": date_indeces,
                          "time_indeces": time_indeces,
                          "chosen_date_indeces": chosen_date_indeces}

            ran_out_of_files, index_dict, flags_dict, counters_dict, all_found_datetimes_adjusted, all_found_time_indeces_adjusted = self._find_indeces_of_matching_datasets(index_dict, flags_dict, 
                                                                                              counters_dict, lowest_resolution_dates, all_found_datetimes_adjusted, all_found_time_indeces_adjusted, 
                                                                                              datasets_datetimes, data_settings_cfgs, datasets_paths, chosen_resolution, max_times_num_per_file)

            if ran_out_of_files:
                warnings.warn('WARNING: Ran out of files with matching datetimes. Number of completed patches may be less than expected. Please consider adjusting "patches_per_unit_time"')
                break

            chosen_date_indeces = index_dict["chosen_date_indeces"]
            date_indeces = index_dict["date_indeces"]
            time_indeces = index_dict["time_indeces"]
            solution_indeces_files = index_dict["solution_indeces_files"]
            solution_indeces_times = index_dict["solution_indeces_times"]
            load_new_files = flags_dict["load_new_files"]
            found_files = flags_dict["found_files"]
            date_counter = counters_dict["date_counter"]
            data_per_file_counter = counters_dict["data_per_file_counter"]

            loaded_datasets, reproj_ds_index = self._load_datasets_from_disk(chosen_date_indeces, solution_indeces_files, solution_indeces_times, datasets_paths, data_settings_cfgs)

            reproj_datasets, dataset_empty_or_out_of_range = self._reproject_datasets(loaded_datasets, reproj_ds_index, data_settings_cfgs)

            # Increment index system for loop around (only relevant for multi-dataset case)
            if data_settings_cfgs[-1]["Data"]["use_internal_times_when_finding_files"]:
                solution_indeces_times[-1] = solution_indeces_times[-1] + 1
            else:
                solution_indeces_files[-1] = solution_indeces_files[-1] + 1
            if data_per_file_counter == max_times_num_per_file:
                load_new_files = True

            if dataset_empty_or_out_of_range:
                warnings.warn('WARNING: At least one of the selected dataset files contained data that was entirely missing or data that did not spatially align with the other datasets. Continuing search...')
                continue

            all_patches, filtered_balanced_counts = self._make_patches(reproj_datasets, data_settings_cfgs, patches_per_time, patch_size, reproj_ds_index, filtered_balanced_counts, number_of_patches_per_balanced_var)

            for single_dataset_patches in all_patches:
                label_patch = None
                feature_patch = None
                for j, patch in enumerate(single_dataset_patches):
                    if data_settings_cfgs[j]["Data"]["is_label_data"]:
                        label_patch = self._merge_patches(label_patch, patch)
                    else:
                        feature_patch = self._merge_patches(feature_patch, patch)
                label_patches = self._concat_patches(label_patches, label_patch)
                feature_patches = self._concat_patches(feature_patches, feature_patch)

        if feature_patches is not None:
            feature_patch_path = os.path.join(feature_patches_root, str(self.run_num) + ".nc")
            feature_patches.to_netcdf(feature_patch_path)
        
        if label_patches is not None:
            label_patch_path = os.path.join(label_patches_root, str(self.run_num) + ".nc")
            label_patches.to_netcdf(label_patch_path)

        print("Completed on search number: " + str(main_loop_counter))
        print("Final patch count per balanced class:")
        print(filtered_balanced_counts)


    def _find_indeces_of_matching_datasets(self, index_dict, flags_dict, counters_dict, lowest_resolution_dates, all_found_datetimes_adjusted, all_found_time_indeces_adjusted,
                                           datasets_datetimes, data_settings_cfgs, datasets_paths, chosen_resolution, max_times_num_per_file):
        # TODO: Turn the packing and unpacking of these dicts into methods (including outside this method)
        chosen_date_indeces = index_dict["chosen_date_indeces"]
        date_indeces = index_dict["date_indeces"]
        time_indeces = index_dict["time_indeces"]
        solution_indeces_files = index_dict["solution_indeces_files"]
        solution_indeces_times = index_dict["solution_indeces_times"]
        load_new_files = flags_dict["load_new_files"]
        found_files = flags_dict["found_files"]
        date_counter = counters_dict["date_counter"]
        data_per_file_counter = counters_dict["data_per_file_counter"]

        if len(chosen_date_indeces) != 1:

            while not found_files:

                while date_counter < len(date_indeces) and load_new_files:
                    data_per_file_counter = 0

                    for i, dataset_datetimes in enumerate(lowest_resolution_dates):

                        dataset_datetimes = np.array(dataset_datetimes)
                        ordered_indeces = np.where(dataset_datetimes == lowest_resolution_dates[-1][date_indeces[date_counter]])[0]
                        random.shuffle(ordered_indeces)
                        chosen_date_indeces[i] = ordered_indeces

                    date_counter = date_counter + 1
                    if np.all([len(i) != 0 for i in chosen_date_indeces]):
                        load_new_files = False
                        solution_indeces_files = np.zeros(len(chosen_date_indeces), dtype=np.int64)
                        solution_indeces_times = np.zeros(len(chosen_date_indeces), dtype=np.int64)

                if date_counter == len(date_indeces) and load_new_files:
                    break

                found_files, solution_indeces_files, solution_indeces_times, all_found_datetimes_adjusted, all_found_time_indeces_adjusted = self._compare_datetimes_with_IO(0,chosen_date_indeces, datasets_datetimes,
                                                                                                                                            data_settings_cfgs,datasets_paths,chosen_resolution,
                                                                                                                                            solution_indeces_files=solution_indeces_files,
                                                                                                                                            solution_indeces_times=solution_indeces_times, 
                                                                                                                                            all_found_datetimes_adjusted=all_found_datetimes_adjusted,
                                                                                                                                            all_found_time_indeces_adjusted=all_found_time_indeces_adjusted)

                if not found_files:
                    load_new_files = True
                else:
                    data_per_file_counter = data_per_file_counter + 1

            if date_counter == len(date_indeces) and load_new_files:
                return True, index_dict, flags_dict, counters_dict, all_found_datetimes_adjusted, all_found_time_indeces_adjusted

            found_files = False

        else:
            if date_counter == len(date_indeces):
                return True, index_dict, flags_dict, counters_dict, all_found_datetimes_adjusted, all_found_time_indeces_adjusted

            solution_indeces_files = [date_indeces[date_counter]]
            solution_indeces_times = [0]
            if data_settings_cfgs[0]["Data"]["has_time_cord"]:
                if data_per_file_counter == 0:
                    y_dim_name = data_settings_cfgs[0]["Data"]["y_dim_name"]
                    x_dim_name = data_settings_cfgs[0]["Data"]["x_dim_name"]
                    ds = self._netcdf_loader(datasets_paths[0][date_indeces[date_counter]], 0, y_dim_name, x_dim_name)
                    time_cord_name = data_settings_cfgs[0]["Data"]["time_cord_name"]
                    current_datetime = ds[time_cord_name]
                    ds.close()

                    time_indeces = np.random.choice(np.arange(0,len(current_datetime)), size=len(current_datetime), replace=False)

                solution_indeces_times = [time_indeces[data_per_file_counter]]

                data_per_file_counter = data_per_file_counter + 1
                if data_per_file_counter == len(time_indeces) or data_per_file_counter == max_times_num_per_file:
                    data_per_file_counter = 0
                    date_counter = date_counter + 1
            else:
                date_counter = date_counter + 1

        index_dict["chosen_date_indeces"] = chosen_date_indeces
        index_dict["date_indeces"] = date_indeces
        index_dict["time_indeces"] = time_indeces
        index_dict["solution_indeces_files"] = solution_indeces_files
        index_dict["solution_indeces_times"] = solution_indeces_times
        flags_dict["load_new_files"] = load_new_files
        flags_dict["found_files"] = found_files
        counters_dict["date_counter"] = date_counter
        counters_dict["data_per_file_counter"] = data_per_file_counter

        return False, index_dict, flags_dict, counters_dict, all_found_datetimes_adjusted, all_found_time_indeces_adjusted

    
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

    
    def _load_datasets_from_disk(self, chosen_date_indeces, solution_indeces_files, solution_indeces_times, datasets_paths, data_settings_cfgs):
        loaded_datasets = []
        reproj_ds_index = -1
        for i, (file_index, time_index) in enumerate(zip(solution_indeces_files, solution_indeces_times)):
            if len(chosen_date_indeces) == 1:
                path = datasets_paths[i][file_index]
            else:
                path = datasets_paths[i][chosen_date_indeces[i][file_index]]
            
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
            if "lon_dim" in dims:
                x_dim_name = "lon_dim"
            else:
                x_dim_name = data_settings_cfgs[i]["Data"]["x_dim_name"]
            if "lat_dim" in dims:
                y_dim_name = "lat_dim"
            else:
                y_dim_name = data_settings_cfgs[i]["Data"]["y_dim_name"]

            # if data_settings_cfgs[i]["Data"]["x_dim_name"] == "lon":
            #     ds = ds.rename_dims({"lon":"lon_dim"})
            #     x_dim_name = "lon_dim"
            # else:
            #     x_dim_name = data_settings_cfgs[i]["Data"]["x_dim_name"]
            # if data_settings_cfgs[i]["Data"]["y_dim_name"] == "lat":
            #     ds = ds.rename_dims({"lat":"lat_dim"})
            #     y_dim_name = "lat_dim"
            # else:
            #     y_dim_name = data_settings_cfgs[i]["Data"]["y_dim_name"]

            # Select only the data we want
            ds = ds[data_settings_cfgs[i]["Data"]["selected_vars"]]

            # If only one key is given ds is reverted to a xarray dataarray.
            # This is not useful for later functions so this is changed back to a xarray dataset
            if type(ds) is xr.core.dataarray.DataArray:
                ds = ds.to_dataset()

            ds = ds.assign_coords(lon=((y_dim_name,x_dim_name), lons))
            ds = ds.assign_coords(lat=((y_dim_name,x_dim_name), lats))

            if data_settings_cfgs[i]["Data"]["reproj_target"]:
                reproj_ds_index = i

            ds = self._select_specific_dims(ds, data_settings_cfgs, i)
            ds = self._add_custom_vars(ds, data_settings_cfgs, i, False)

            loaded_datasets.append(ds)

        if reproj_ds_index == -1:
            raise Exception("No dataset has been designated as the reproj_target. You must designate exactly one as this. Even if only loading one dataset.")

        return loaded_datasets, reproj_ds_index


    def _reproject_datasets(self, loaded_datasets, reproj_ds_index, data_settings_cfgs):
        reproj_datasets = []
        dataset_empty_or_out_of_range = False
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
                        dataset_empty_or_out_of_range = True
                        break

                if dataset_empty_or_out_of_range:
                    break
            else:
                ds_reproj = ds

            ds_reproj = self._add_custom_vars(ds_reproj, data_settings_cfgs, i, True)

            reproj_datasets.append(ds_reproj)

            if data_settings_cfgs[i]["Data"]["needs_reproj"]:
                ds.close()

        return reproj_datasets, dataset_empty_or_out_of_range

    
    # Reshapes numpy arrays from (patch_row_num, patch_col_num, n_variables_across_all_datasets, patch_dim_0, patch_dim_1)
    # to (n_patches, n_variables_across_all_datasets, n_pixels_per_patch). See use below.
    def _np_reshaper(self, master_np_array):
        list_ds_shape = list(master_np_array.shape)
        patch_dim0 =  list_ds_shape.pop(3)
        patch_dim1 = list_ds_shape[3]
        list_ds_shape[3] = patch_dim1*patch_dim0
        patch_index0 = list_ds_shape.pop(0)
        patch_index1 = list_ds_shape[0]
        list_ds_shape[0] = patch_index0*patch_index1

        return master_np_array.reshape(list_ds_shape)


    # Use numpy broadcasting (memory efficient) to create one large numpy out of all datasets.
    # Used for run efficient mass modifications and filters on all patches. (FAR more effective than loops)
    def _make_master_np_array(self, reproj_datasets, x_dim_name, y_dim_name, num_dims_to_concat=3):
        reproj_dataarrays_broadcasted = []
        new_array_shape = ()
        new_np_array_dim_keys = ()
        data_arrays_dim_mappings = []
        all_new_keys = [] # Dimension keys in final broadcasted form
        reproj_dataarrays = []
        all_keys = [] # This is dimension keys, not var keys
        all_var_keys = [] # This is var keys
        dataset_names = self.dataset_names

        for i, ds in enumerate(reproj_datasets):
            da = ds.to_array()
            all_keys.append(list(ds.dims))
            if num_dims_to_concat == 4:
                da = da.transpose("variable", x_dim_name, y_dim_name, "time_dim", ...)
            else:
                da = da.transpose("variable", x_dim_name, y_dim_name, ...)

            reproj_dataarrays.append(da)
            all_var_keys.append(list(ds.keys()))

        all_var_keys = np.concatenate(all_var_keys, axis=0)

        for j, reproj_dataarray in enumerate(reproj_dataarrays):
            new_keys = [dataset_names[j]+"_"+key for key in all_keys[j][num_dims_to_concat:]]
            dim_mapping = {key:ind+num_dims_to_concat for ind, key in enumerate(new_keys)}
            data_arrays_dim_mappings.append(dim_mapping)
            all_new_keys.append(new_keys)
            new_array_shape = new_array_shape + reproj_dataarray.shape[num_dims_to_concat:]
            new_np_array_dim_keys = new_np_array_dim_keys + tuple(dim_mapping.keys())

        new_np_array_dim_mappings = {key:ind+num_dims_to_concat for ind, key in enumerate(new_np_array_dim_keys)}

        # TODO: Reduce memory consumption with less names here?
        for j, reproj_dataarray in enumerate(reproj_dataarrays):
            reproj_dataarray_expanded = np.expand_dims(reproj_dataarray, tuple(np.arange(len(reproj_dataarray.shape),len(new_array_shape)+num_dims_to_concat)))
            if len(all_new_keys[j]) != 0:
                reproj_dataarray_expanded = np.moveaxis(reproj_dataarray_expanded, itemgetter(*all_new_keys[j])(data_arrays_dim_mappings[j]), 
                                                                            itemgetter(*all_new_keys[j])(new_np_array_dim_mappings))
            reproj_dataarray_broadcasted = np.broadcast_to(reproj_dataarray_expanded, reproj_dataarray_expanded.shape[:num_dims_to_concat] + new_array_shape)
            reproj_dataarrays_broadcasted.append(reproj_dataarray_broadcasted)

        return np.concatenate(reproj_dataarrays_broadcasted, axis=0), all_var_keys, all_new_keys


    def _filter_patch_pixels(self, reproj_datasets, patch_size, x_dim_name, y_dim_name, data_settings_cfgs, patches_per_time, shuffle, filter_using_time):
        filtered_balanced_pixels = []
        all_filters = []
        all_thresholds = []
        all_balanced_filters = []
        all_balanced_thresholds = []
        for data_settings_cfg in data_settings_cfgs:
            for i, filter in enumerate(data_settings_cfg["Filtration"]["filters"]):
                all_filters.append(filter)
                all_thresholds.append(data_settings_cfg["Filtration"]["filter_patch_threshold"][i])
            for i, filter in enumerate(data_settings_cfg["Filtration"]["filters_balanced"]):
                all_balanced_filters.append(filter)
                all_balanced_thresholds.append(data_settings_cfg["Filtration"]["filter_patch_threshold_balanced"][i])

        if len(all_balanced_filters) == 0:
            num_of_filters = 1
        else:
            num_of_filters = len(all_balanced_filters)

        # NOTE: This new process assumes that any time dimensions have been stacked together in each individual xarray dataset object

        # Determine how many dims to concat based on if the time dim is present
        if filter_using_time:
            num_master_array_dims_to_concat = 4
        else:
            num_master_array_dims_to_concat = 3

        # TODO: Make this return dicts instead of arrays for the keys
        # Creates a broadcasted numpy array of all our data across all dimensions for powerful manipulation
        dataset_master_np_array, var_keys, dim_keys = self._make_master_np_array(reproj_datasets, x_dim_name, y_dim_name, num_master_array_dims_to_concat)

        # If domains do not nicely line up with factors of patch size, cut out extra
        if dataset_master_np_array.shape[1] % patch_size != 0:
            extra_pixels = dataset_master_np_array.shape[1] % patch_size
            dataset_master_np_array = np.take(dataset_master_np_array, np.arange(dataset_master_np_array.shape[1]-extra_pixels), axis=1)
        if dataset_master_np_array.shape[2] % patch_size != 0:
            extra_pixels = dataset_master_np_array.shape[2] % patch_size
            dataset_master_np_array = np.take(dataset_master_np_array, np.arange(dataset_master_np_array.shape[2]-extra_pixels), axis=2)

        # Make array for tracking indeces in original dataset of each new patch
        ds_dims_x, ds_dims_y = np.meshgrid(np.arange(dataset_master_np_array.shape[1]), np.arange(dataset_master_np_array.shape[2]))
        ds_dims = np.stack([ds_dims_y, ds_dims_x], axis=0)

        # Setup bool array for tracking patch rejection
        vaid_pixels_bool = np.zeros((dataset_master_np_array.shape[1], dataset_master_np_array.shape[2], num_of_filters))

        # Split our domains into evenly distributed patches
        # shape of (patch_row_num, patch_col_num, n_variables_across_all_datasets, patch_dim_0, patch_dim_1, ...)
        # and (patch_row_num, patch_col_num, ds_dims, patch_dim_0, patch_dim_1) respectively
        dataset_master_np_array = np.array(np.split(dataset_master_np_array, dataset_master_np_array.shape[2]/patch_size, axis=2))
        dataset_master_np_array = np.array(np.split(dataset_master_np_array, dataset_master_np_array.shape[2]/patch_size, axis=2))
        ds_dims = np.array(np.split(ds_dims, ds_dims.shape[2]/patch_size, axis=2))
        ds_dims = np.array(np.split(ds_dims, ds_dims.shape[2]/patch_size, axis=2))

        # Reshape to (n_patches, n_variables_across_all_datasets, n_pixels_per_patch, ...)
        # and (n_patches, ds_dims, n_pixels_per_patch) respectively
        # NOTE: This may be removed later if it is deemed better to keep the other shape
        dataset_master_np_array = self._np_reshaper(dataset_master_np_array)
        ds_dims = self._np_reshaper(ds_dims)

        # Warn if you are trying to request more patches per unit time than is possible
        if dataset_master_np_array.shape[0] < patches_per_time:
            warnings.warn('At least one collection of datasets at a particular search attempt does not have enough space for your given "patches_per_time". Consider using less?')

        # Select the indeces for the upperleft corners of each patch only. (for the patch making function)
        ds_dims = ds_dims[:,:,0]

        # Init all patch corners of boolean array to 1. Our filters switch them to 0 if they don't meet our requirements
        vaid_pixels_bool[ds_dims[:,0], ds_dims[:,1], :] = 1

        # Mark for removal any patches that contain any nans anywhere and in any dimension
        indeces_of_nans = np.isnan(dataset_master_np_array)
        indeces_of_nan_test_failure = np.nonzero(np.any(indeces_of_nans, axis=tuple(np.arange(1,len(dataset_master_np_array.shape)))))
        vaid_pixels_bool[ds_dims[indeces_of_nan_test_failure[0],0], ds_dims[indeces_of_nan_test_failure[0],1], :] = 0

        # Mark for removal any patches that fail our broad filters
        for j, filter in enumerate(all_filters):
            return_dict = OrderedDict()
            exec(filter, globals().update(locals()), return_dict)
            return_list = list(return_dict.items())
            sum_of_each_patch = return_list[-1][-1]
            rejected_pixels = np.nonzero(sum_of_each_patch < all_thresholds[j])
            vaid_pixels_bool[ds_dims[rejected_pixels[0],0], ds_dims[rejected_pixels[0],1], :] = 0

        # Categorize patches by whether or not they fail our balanced filters
        for j, filter in enumerate(all_balanced_filters):
            return_dict = OrderedDict()
            exec(filter, globals().update(locals()), return_dict)
            return_list = list(return_dict.items())
            sum_of_each_patch = return_list[-1][-1]
            rejected_pixels = np.nonzero(sum_of_each_patch < all_balanced_thresholds[j])
            vaid_pixels_bool[ds_dims[rejected_pixels[0],0], ds_dims[rejected_pixels[0],1], j] = 0
        
        # TODO: Consider if this legacy code block should be replaced. Calling np.where again unnecessary?
        for i in range(num_of_filters):
            where_array = np.array(np.where(vaid_pixels_bool[:,:,i] == 1))
            # TODO: Check this np.size if statement with some tests because I don't fully trust it
            if np.size(where_array):
                if shuffle:
                    combined_lists_for_shuffle = list(zip(where_array[0], where_array[1]))
                    random.shuffle(combined_lists_for_shuffle)
                    where_array_x, where_array_y = zip(*combined_lists_for_shuffle)
                    where_array_x, where_array_y = np.array(where_array_x), np.array(where_array_y)
                    filtered_balanced_pixels.append(np.array([where_array_x, where_array_y]))
                else:
                    filtered_balanced_pixels.append(where_array)
            else:
                filtered_balanced_pixels.append(np.array([[], []]))

        return filtered_balanced_pixels

    
    def _make_patches(self, reproj_datasets, data_settings_cfgs, patches_per_time, patch_size, reproj_ds_index, filtered_balanced_counts, number_of_patches_per_balanced_var):
        reproj_ds = reproj_datasets[reproj_ds_index]

        dims = [dim for dim in reproj_ds.dims]
        if "lon_dim" in dims:
            x_dim_name = "lon_dim"
        else:
            x_dim_name = data_settings_cfgs[reproj_ds_index]["Data"]["x_dim_name"]
        if "lat_dim" in dims:
            y_dim_name = "lat_dim"
        else:
            y_dim_name = data_settings_cfgs[reproj_ds_index]["Data"]["y_dim_name"]

        all_patches = []
        pixel_counters = np.zeros(len(filtered_balanced_counts), dtype=np.int64)

        # TODO: Implement shuffle here
        # TODO: Implement time stuff here or elsewhere?
        filtered_balanced_pixels = self._filter_patch_pixels(reproj_datasets,patch_size,x_dim_name,y_dim_name,data_settings_cfgs, patches_per_time, self.shuffle_patches_in_each_timestep, False)
        
        for i in range(patches_per_time):
            single_dataset_patches = []
            filter_balance_order = np.array(filtered_balanced_counts).argsort()
            for filter_balance_ind in filter_balance_order:
                if filtered_balanced_counts[filter_balance_ind] < number_of_patches_per_balanced_var and len(filtered_balanced_pixels[filter_balance_ind][0]) > 0 and pixel_counters[filter_balance_ind] < len(filtered_balanced_pixels[filter_balance_ind][0]):
                    x_i = filtered_balanced_pixels[filter_balance_ind][0][pixel_counters[filter_balance_ind]]
                    y_i = filtered_balanced_pixels[filter_balance_ind][1][pixel_counters[filter_balance_ind]]
                    
                    for ds in reproj_datasets:
                        patch = self._make_patch(ds, patch_size, x_i, y_i, x_dim_name, y_dim_name)
                        single_dataset_patches.append(patch)

                    filtered_balanced_counts[filter_balance_ind] = filtered_balanced_counts[filter_balance_ind] + 1
                    pixel_counters[filter_balance_ind] = pixel_counters[filter_balance_ind] + 1
                    break

            if len(single_dataset_patches) != 0:
                all_patches.append(single_dataset_patches)

        if np.sum(pixel_counters) < patches_per_time:
            warnings.warn("While generating patches for a single timestep, the function _make_patches ran out of possible patches that meet the set filters' requirements. Continuing search...")

        print("Accepted Patches: " + str(pixel_counters))

        for ds in reproj_datasets:
            ds.close()

        return all_patches, filtered_balanced_counts

    
    def _merge_patches(self, patches, patch):
        if patches is None:
            patches = copy.deepcopy(patch)
            patch.close()
        else:
            patches = patches.merge(patch)

        return patches

    
    def _concat_patches(self, patches, patch):
        if patches is None:
            patches = copy.deepcopy(patch.expand_dims(dim='n_samples'))
            patch.close()
        else:
            patches = xr.concat([patches,patch.expand_dims(dim='n_samples')],dim='n_samples')

        return patches
    

    def _make_patch(self, file_ds, patch_size, x, y, x_dim_name, y_dim_name):
        patch = file_ds[{x_dim_name:slice(x, x+patch_size), y_dim_name:slice(y, y+patch_size)}]

        return patch


    # NOTE: Assumes np_array is array (NOT LIST) of np.datetime64 objects (NOT datetime.date objects)
    def _convert_datetime64_array_to_list(self, np_array):
        new_list = []
        for np_datetime in np_array:
            new_list.append(np_datetime)
        return new_list

    # Three possible options:
    # 1. use dates to set range
    # 2. use hard indeces to set range
    # 3. use percentages (real numbers) to set range
    # NOTE: Does not check validity of selections. May add that later
    # NOTE: Please make sure the start and end dates in ISO_8601 format
    def select_data_range(self, file_list, data_start, data_end, use_date=False, dates_list=None, dataset_date_resolution=None):
        file_list = np.array(file_list)

        # If date is used for selecting files, this assumes that the dates are included and are not None TODO: Make error check for this?
        # (Just throw exception inside if statement below that checks if dates_list is none)
        if use_date:
            if data_start is not None:
                start_date = np.datetime64(data_start).astype(dataset_date_resolution)
            if data_end is not None:
                end_date = np.datetime64(data_end).astype(dataset_date_resolution)

            if data_start is not None and data_end is not None:
                inds = np.where(np.logical_and(dates_list >= start_date, dates_list <= end_date))
                dates_list = np.array(dates_list)
                return file_list[inds].tolist(), self._convert_datetime64_array_to_list(dates_list[inds])
            elif data_start is None and data_end is not None:
                inds = np.where(dates_list <= end_date)
                dates_list = np.array(dates_list)
                return file_list[inds].tolist(), self._convert_datetime64_array_to_list(dates_list[inds])
            elif data_start is not None and data_end is None:
                inds = np.where(dates_list >= start_date)
                dates_list = np.array(dates_list)
                return file_list[inds].tolist(), self._convert_datetime64_array_to_list(dates_list[inds])
            else:
                return file_list.tolist(), dates_list
        
        else:
            start_index = 0
            end_index = -1

            if data_start is not None:
                if isinstance(data_start, int):
                    start_index = data_start
                elif isinstance(data_start, float):
                    start_index = int(data_start*len(file_list))

            if data_end is not None:
                if isinstance(data_end, int):
                    end_index = data_end
                elif isinstance(data_end, float):
                    end_index = int(data_end*len(file_list))
        
            if dates_list is not None:
                dates_list = np.array(dates_list)
                dates_list = self._convert_datetime64_array_to_list(dates_list[start_index:end_index])

            return file_list[start_index:end_index].tolist(), dates_list


    # Extracts the best possible time information from data's directory structure and file name 
    # NOTE: See the required characters for designating where each datetime component is located in each file's path
    # NOTE: If a lower level datetime unit is used (for example hours or minute), every higher level must also be present
    # NOTE: datetime_positions chars differ from the datetime.datetime chars needed for "datetime_ISO_formats". May change this later
    # NOTE: In the path/name of each file there must be at least SOME datetime information. The no-information scenario is not currently supported
    def extract_best_datetime_no_IO(self, root_path, data_file_list, datetime_positions, extraction_regs, datetime_ISO_formats):
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

        # TODO: Double check datetime64[char] have been chosen right
        if "m" in datetime_chars_seperated:
            dataset_date_resolution = "datetime64[m]"
            dataset_date_resolution_val = 4
        elif "h" in datetime_chars_seperated:
            dataset_date_resolution = "datetime64[h]"
            dataset_date_resolution_val = 3
        elif "D" in datetime_chars_seperated:
            dataset_date_resolution = "datetime64[D]"
            dataset_date_resolution_val = 2
        elif "M" in datetime_chars_seperated:
            dataset_date_resolution = "datetime64[M]"
            dataset_date_resolution_val = 1
        else:
            dataset_date_resolution = "datetime64[Y]"
            dataset_date_resolution_val = 0

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

        return datetimes, dataset_date_resolution, dataset_date_resolution_val


    # TODO: May not need this in a function anymore???
    def _glob_path_maker(self, root_path, glob_string):
        return root_path.rstrip("/") + "/" + glob_string.lstrip("/")


    # NOTE: In path_glob only include wild card operators for each directory level you want to search across.
    #       The regex can handle any filtering.
    def create_file_list(self, root_dir, path_glob, path_regex):
        glob_path = self._glob_path_maker(root_dir, path_glob)
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

        # TODO: Make the program crash if file_list stays empty. (ALSO DO THIS IN THE TIME CHECKING FUNCTIONS TOO)
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
    patcher = Patcher(args.run_num)
    config = configparser.ConfigParser()
    config.read(args.config_path)
    config = cfg_parser(config)
    patcher.run(config)

# TODO: REMINDER: WoFS file and data names change all the time. Look out for this!