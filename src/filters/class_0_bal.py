from filters.bin_class_bal import bin_class_bal

THIS_FILTER_RETURNS_PATCH_MASK = True

def class_0_bal(ds, maximized_dims, patch_size):
    return bin_class_bal(ds, maximized_dims, patch_size, "data_var == 0")