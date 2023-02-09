import numpy as np

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
    "scp": [0, 100],
    "cape_sfc": [0, 20000],
    "cin_sfc": [-2000, 0],
    "lfc_mu": [0, 10000],
    "lcl_mu": [0, 10000],
    "MESH95": [0, 254],
    "MESH_class_bin": [0,1]
}

def wofs_bounds(ds, maximized_dims):
    lat_len = ds.dims["lat_dim"]
    lon_len = ds.dims["lon_dim"]
    valid_pixels_shape = (lon_len, lat_len)
    valid_pixels = np.ones(valid_pixels_shape, dtype=np.int64)
    ds = ds.transpose("lat_dim", ...)
    ds = ds.transpose("lon_dim", ...)

    for key in ds.keys():
        data_var = ds[key].to_numpy()
        data_var_shape = data_var.shape
        bad_pixel_bool = np.logical_or(data_var < field_bounds[key][0], data_var > field_bounds[key][1])
        if len(data_var_shape) > 2:
            bad_pixel_bool = np.any(bad_pixel_bool, axis=np.arange(2,len(data_var_shape)))
        valid_pixels[np.nonzero(bad_pixel_bool)] = 0
    
    return valid_pixels