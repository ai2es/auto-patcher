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
    "scp": [-100, 100],
    "cape_sfc": [0, 20000],
    "cin_sfc": [-2000, 0],
    "lfc_mu": [0, 82021],
    "lcl_mu": [0, 82021],
    "MESH95": [0, 254],
    "MESH_class_bin": [0,1]
}

# NOTE: This may crash if dataset with only one data variable is passed in
def wofs_bounds(ds, maximized_dims):
    ds = ds.transpose("lat_dim", ...)
    ds = ds.transpose("lon_dim", ...)

    lat_len = ds.dims["lat_dim"]
    lon_len = ds.dims["lon_dim"]
    valid_pixels_shape = [lon_len, lat_len]
    for dim_name in maximized_dims:
        valid_pixels_shape.append(ds.dims[dim_name])
    valid_pixels_shape = tuple(valid_pixels_shape)
    valid_pixels = np.ones(valid_pixels_shape, dtype=np.int64)

    for key in ds.keys():
        data_var = ds[key]

        ds_dims = list(data_var.dims)
        ds_dims = np.array(ds_dims)

        ds_dims_to_collapse = list(data_var.dims)
        ds_dims_to_collapse.remove("lat_dim")
        ds_dims_to_collapse.remove("lon_dim")
        for dim_name in maximized_dims:
            ds_dims_to_collapse.remove(dim_name)
        ds_dims_to_collapse = np.array(ds_dims_to_collapse)

        ds_dims_to_collapse = np.in1d(ds_dims, ds_dims_to_collapse).nonzero()[0]

        bad_pixel_bool = np.logical_or(data_var.to_numpy() < field_bounds[key][0], data_var.to_numpy() > field_bounds[key][1])
        bad_pixel_bool = np.any(bad_pixel_bool, axis=tuple(ds_dims_to_collapse))
        valid_pixels[np.nonzero(bad_pixel_bool)] = 0
    
    return valid_pixels