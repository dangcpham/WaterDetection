import numpy as np

# basic directory paths
DATA_DIR = "data"
OUTPUT_DIR = "output"

# wavelength info
WAVELENGTH_RESOLUTION = 10000
WAVELENGTH_MAX = 2.35
WAVELENGTH_MIN = 0.41

# all the surfaces
cloud_path = f"{DATA_DIR}/cloud/cloud_all.csv"
snow_path = "{DATA_DIR}/melting_snow_1-16.9366.asc"
sand_path = f"{DATA_DIR}/quartz_gds74.5830.asc"
seawater_path = f"{DATA_DIR}/seawater_open_ocean_sw2.9627.asc"
basalt_path = f"{DATA_DIR}/basalt_weathered_br93-43.7492.asc"
veg_path = f"{DATA_DIR}/Agrococcus_KM349956.csv"

surface_paths = {"snow": snow_path, "sand": sand_path,
                 "seawater": seawater_path, "basalt": basalt_path,
                 "veg": veg_path}
cloud_path = [cloud_path]

# resolution of composition, for each component
DELTA_STEP = 5

# fake filters info
FILTER_SIZE = 0.2 # in micron
FILTER_MIN  = 0.45 # starting wavelength
FILTER_MAX  = 2.35 # final wavelength

# params for ML classifications
CLASSIFYING_COMPONENTS = ['seawater','snow','cloud']
SNRS = np.arange(5, 105, 10)
RANDOM_INITIALIZATIONS = 100