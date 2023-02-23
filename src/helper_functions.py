import numpy as np

class Kernelizer:
    """
    Conerts the 1D function defined by func parameter in the constructor to a striding kernel problem using vectorized code.
    In layman terms this will run func (usually a numpy func) as a 2D kernel with size of patch_size and stride of 1 across 
    the first two axes of array paramter in kernelize(). Vectorization allows for low runtimes. func must take a 1d array as a parameter.
    May allow greater strides in the future but this is not supported right now.
    """
    def __init__(self, func, patch_size):
        self.func = func
        self.patch_size = patch_size

    def _kernel_along_one_element(self, array_1d):
        for i in range(len(array_1d)-self.patch_size):
            array_1d[i] = self.func(array_1d[i:i+self.patch_size])
        return array_1d

    def _kernel_along_axis(self, array_2d, axis):
        return np.apply_along_axis(self._kernel_along_one_element, axis, array_2d)

    def kernelize(self, array):
        return np.apply_over_axes(self._kernel_along_axis, array, axes=[0,1])


def setup_valid_pixel_array(ds, maximized_dims, init_as_ones=False):
    lat_len = ds.dims["lat_dim"]
    lon_len = ds.dims["lon_dim"]

    valid_pixels_shape = [lon_len, lat_len]
    for dim_name in maximized_dims:
        valid_pixels_shape.append(ds.dims[dim_name])
    valid_pixels_shape = tuple(valid_pixels_shape)

    if init_as_ones:
        valid_pixels = np.ones(valid_pixels_shape, dtype=np.int64)
    else:
        valid_pixels = np.zeros(valid_pixels_shape, dtype=np.int64)
    
    return valid_pixels


def get_indices_of_dims_to_collapse(ds, maximized_dims):
    ds_dims = list(ds.dims)
    ds_dims = np.array(ds_dims)

    ds_dims_to_collapse = list(ds.dims)
    ds_dims_to_collapse.remove("lat_dim")
    ds_dims_to_collapse.remove("lon_dim")
    for dim_name in maximized_dims:
        ds_dims_to_collapse.remove(dim_name)
    ds_dims_to_collapse = np.array(ds_dims_to_collapse)

    ds_dims_to_collapse = np.in1d(ds_dims, ds_dims_to_collapse).nonzero()[0]

    return ds_dims_to_collapse