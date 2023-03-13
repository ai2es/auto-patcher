from helper_functions import Kernelizer, setup_valid_pixel_array, get_indices_of_dims_to_collapse
import numpy as np

THIS_FILTER_RETURNS_PATCH_MASK = True
BIN_CLASS_NAME = "MESH_class_bin"

def bin_class_bal(ds, maximized_dims, patch_size, threshold_code_str):
    ds = ds.transpose("lat_dim", ...)
    ds = ds.transpose("lon_dim", ...)

    valid_pixels = setup_valid_pixel_array(ds, maximized_dims, False)

    data_var = ds[BIN_CLASS_NAME]

    ds_dims_to_collapse = get_indices_of_dims_to_collapse(data_var, maximized_dims)

    data_var = np.sum(data_var, axis=tuple(ds_dims_to_collapse))
    kernelizer = Kernelizer(np.sum, patch_size)
    data_var = kernelizer.kernelize(data_var)

    filter_mask = eval("np.nonzero(" + threshold_code_str + ")")
    valid_pixels[filter_mask] = 1
    
    return valid_pixels