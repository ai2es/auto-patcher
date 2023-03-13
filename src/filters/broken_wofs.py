import numpy as np
from wofs_bounds import field_bounds
from helper_functions import setup_valid_pixel_array, get_indices_of_dims_to_collapse

THIS_FILTER_RETURNS_PATCH_MASK = False

def broken_wofs(ds, maximized_dims, patch_size):
    ds = ds.transpose("lat_dim", ...)
    ds = ds.transpose("lon_dim", ...)

    bad_pixels = setup_valid_pixel_array(ds, maximized_dims, False)

    for key in ds.keys():
        data_var = ds[key]

        ds_dims_to_collapse = get_indices_of_dims_to_collapse(ds, maximized_dims)

        bad_pixel_bool = np.logical_or(data_var.to_numpy() < field_bounds[key][0], data_var.to_numpy() > field_bounds[key][1])
        bad_pixel_bool = np.any(bad_pixel_bool, axis=tuple(ds_dims_to_collapse))
        bad_pixels[np.nonzero(bad_pixel_bool)] = 1
    
    return bad_pixels