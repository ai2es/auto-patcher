import numpy as np
from operator import itemgetter

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