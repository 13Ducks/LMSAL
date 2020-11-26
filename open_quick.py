from iris_lmsalpy import extract_irisL2data

raster_filename = "iris_l2_20160114_230409_3630008076_raster_t000_r00000.fits"
iris_raster = extract_irisL2data.load(
    raster_filename, window_info=["Mg II k 2796"], verbose=True
)

iris_raster.quick_look()
del iris_raster