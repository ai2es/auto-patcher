import numpy as np
from wofs_bounds import field_bounds

def broken_wofs(ds, maximized_dims):
    lat_len = ds.dims["lat_dim"]
    lon_len = ds.dims["lon_dim"]
    bad_pixels_shape = (lon_len, lat_len)
    bad_pixels = np.zeros(bad_pixels_shape, dtype=np.int64)
    ds = ds.transpose("lat_dim", ...)
    ds = ds.transpose("lon_dim", ...)

    for key in ds.keys():
        data_var = ds[key].to_numpy()
        data_var_shape = data_var.shape
        bad_pixel_bool = np.logical_or(data_var < field_bounds[key][0], data_var > field_bounds[key][1])
        if len(data_var_shape) > 2:
            bad_pixel_bool = np.any(bad_pixel_bool, axis=tuple(np.arange(2,len(data_var_shape))))
        bad_pixels[np.nonzero(bad_pixel_bool)] = 1
    
    return bad_pixels