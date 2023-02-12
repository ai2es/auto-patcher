import numpy as np
from wofs_bounds import field_bounds

def broken_wofs(ds, maximized_dims):
    ds = ds.transpose("lat_dim", ...)
    ds = ds.transpose("lon_dim", ...)

    da = ds.to_array()

    ds_dims = list(da.dims)
    ds_dims.remove("variable")
    ds_dims = np.array(ds_dims)

    ds_dims_to_collapse = list(da.dims)
    ds_dims_to_collapse.remove("lat_dim")
    ds_dims_to_collapse.remove("lon_dim")
    ds_dims_to_collapse.remove("variable")
    for dim_name in maximized_dims:
        ds_dims_to_collapse.remove(dim_name)
    ds_dims_to_collapse = np.array(ds_dims_to_collapse)

    ds_dims_to_collapse = np.in1d(ds_dims, ds_dims_to_collapse).nonzero()[0]

    lat_len = ds.dims["lat_dim"]
    lon_len = ds.dims["lon_dim"]
    valid_pixels_shape = [lon_len, lat_len]
    for dim_name in maximized_dims:
        valid_pixels_shape.append(ds.dims[dim_name])
    valid_pixels_shape = tuple(valid_pixels_shape)
    bad_pixels = np.zeros(valid_pixels_shape, dtype=np.int64)

    for key in ds.keys():
        data_var = ds[key].to_numpy()
        bad_pixel_bool = np.logical_or(data_var < field_bounds[key][0], data_var > field_bounds[key][1])
        bad_pixel_bool = np.any(bad_pixel_bool, axis=tuple(ds_dims_to_collapse))
        bad_pixels[np.nonzero(bad_pixel_bool)] = 1
    
    return bad_pixels