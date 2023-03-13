import numpy as np
from helper_functions import setup_valid_pixel_array, get_indices_of_dims_to_collapse

THIS_FILTER_RETURNS_PATCH_MASK = False

field_bounds = {
    "uh_2to5": [0, 5000],
    "w_up": [-10, 100],
    "freezing_level": [0, 10000],
    "hail": [0, 10],
    "hailcast": [0, 10],
    "comp_dz": [-20, 90],
    "td_2": [-100, 100],
    "cape_mu":  [0, 20000],
    "cin_mu": [-2000, 0],
    "shear_v_0to6": [-200, 200],
    "shear_u_0to6": [-200, 200],
    "srh_0to1": [-2000, 5000],
    "srh_0to3": [-2000, 5000],
    "scp": [-100, 100],
    "cape_sfc": [0, 20000],
    "cin_sfc": [-2000, 0],
    "lfc_mu": [0, 82021],
    "lcl_mu": [0, 82021],
    "MESH95": [0, 254],
    "MESH_class_bin": [0,1]
}

# NOTE: This may crash if dataset with only one data variable is passed in
def wofs_bounds(ds, maximized_dims, patch_size):
    ds = ds.transpose("lat_dim", ...)
    ds = ds.transpose("lon_dim", ...)

    valid_pixels = setup_valid_pixel_array(ds, maximized_dims, True)

    for key in ds.keys():
        data_var = ds[key]

        ds_dims_to_collapse = get_indices_of_dims_to_collapse(data_var, maximized_dims)

        bad_pixel_bool = np.logical_or(data_var.to_numpy() < field_bounds[key][0], data_var.to_numpy() > field_bounds[key][1])
        bad_pixel_bool = np.any(bad_pixel_bool, axis=tuple(ds_dims_to_collapse))
        valid_pixels[np.nonzero(bad_pixel_bool)] = 0
    
    return valid_pixels