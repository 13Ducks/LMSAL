from iris_lmsalpy import extract_irisL2data, saveall as sv

raster_filename = "/Users/aaryan/Documents/Code/LMSAL_HUB/iris_hub/iris_l2_20160908_123116_3893010094_raster_t000_r00000.fits"
iris_raster = extract_irisL2data.load(
    raster_filename, window_info=["Mg II k 2796"], verbose=True
)

#aux = sv.load(raster_filename[:-5]+"/kmeans_data.jbl.gz")

#iris_raster.copy_raster("Mg II k 2796", "k-means")
#iris_raster.raster["k-means"].data = aux['kmap']
#iris_raster.raster["Mg II k 2796"].overplot = True
#iris_raster.raster["Mg II k 2796"].over_1 = "k-means"

iris_raster.quick_look()
del iris_raster, aux
