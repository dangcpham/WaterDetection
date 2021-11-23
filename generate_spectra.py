import numpy as np
from scipy.interpolate import interp1d

import pickle
import pandas as pd
from utils import get_surface_data, make_all_spectra
import settings

################################## LOAD PREREQ DATA ###############################

X = np.linspace(settings.WAVELENGTH_MIN, settings.WAVELENGTH_MAX, 
                settings.WAVELENGTH_RESOLUTION)

# get sun transmissions
sun_transmission = get_surface_data([f"{settings.DATA_DIR}/cold_sun.csv"], X)[0]

# get earth transmission to surface 
earth_transmission_path = f"{settings.DATA_DIR}/Earth_Sun0.875_Surface"
wavelength, albedo = np.genfromtxt(earth_transmission_path, dtype=None, unpack=True)

earth_interp = interp1d(wavelength, albedo)
earth_transmission = earth_interp(X)*sun_transmission

# get pressure at 6km
atmosphere_data_path = f"{settings.DATA_DIR}/clima_ColdEarth_Sun0.875"
atmosphere_data = ascii.read(atmosphere_data_path)
pressure_interp = interp1d(atmosphere_data['ALT'], atmosphere_data['P'])
pressure_6km = pressure_interp(6)

# get 6km cloud albedo
path_6km = f"{settings.DATA_DIR}/Earth_Sun0.875_6kmCloud"
wavelength, albedo = np.genfromtxt(path_6km, dtype=None, unpack=True)
interp_6km = interp1d(wavelength, albedo)
albedo_6km = interp_6km(X)
transmission_6km = sun_transmission*albedo_6km

# get surface data
surface_data = get_surface_data(settings.surface_paths.values(), X)
cloud_data = get_surface_data(settings.cloud_path, X)

# load all component combinations that sum to unity
# (output from generate_combinations.py)
component_names, unity_surface_combinations = pickle.load(
        open(f"{settings.OUTPUT_DIR}/surface_combinations.pkl", "rb"))

################################## MAKE SPECTRA ###############################

# make spectra
all_spectra = make_all_spectra(X, combinations = unity_surface_combinations, 
                                cloud_index = component_names.index("cloud"),
                                surfaces_albedo = surface_data, 
                                cloud_albedo = cloud_data,
                                transmission_surface = earth_transmission,
                                transmission_6km = transmission_6km,
                                p_6km = pressure_6km,
                                star_spectrum = sun_transmission)

# save spectra data
pickle.dump(all_spectra, open("output/all_spectra.pkl", "wb"))

