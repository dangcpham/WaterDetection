import numpy as np
from scipy.interpolate import interp1d
import pickle
from astropy.io import ascii

import utils
import settings

################################## LOAD PREREQ DATA ############################

X = np.linspace(settings.WAVELENGTH_MIN, settings.WAVELENGTH_MAX, 
                settings.WAVELENGTH_RESOLUTION)

# get sun transmissions
sun_transmission = utils.get_surface_data(
    [f"{settings.DATA_DIR}/cold_sun.csv"], X)[0]

# get earth transmission to surface 
earth_transmission_path = f"{settings.DATA_DIR}/Earth_Sun0.875_Surface"
wavelength, albedo = np.genfromtxt(earth_transmission_path, dtype=None, 
                                   unpack=True)

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
surface_data = utils.get_surface_data(settings.surface_paths.values(), X)
cloud_data = utils.get_surface_data(settings.cloud_path, X)

# load good filters
good_filters_info = pickle.load(open("output/good_filters_info.pkl", "rb"))
good_filters_names = list(good_filters_info.keys())
good_filters_values = np.array(list(good_filters_info.values()))

##################### SETUP MODEL AND PROBABILITIES ############################

def eval_model(theta):
    """
    Create colors from good filters list and composition theta.

    Parameters:
        theta (array): combinations list (composition)

    Returns:
         (array): list of colors at filters
    """
    theta = np.array([theta])
    flux = utils.make_all_spectra(X, combinations = theta, 
                                cloud_index = 0,
                                surfaces_albedo = surface_data, 
                                cloud_albedo = cloud_data,
                                transmission_surface = earth_transmission,
                                transmission_6km = transmission_6km,
                                p_6km = pressure_6km,
                                star_spectrum = sun_transmission)
    
    colors = np.zeros((len(good_filters_values), 1))
    
    for i, filter_val in enumerate(good_filters_values):
        filter_response = flux * utils.filter_func(filter_val, X, 
                                            step=settings.FILTER_SIZE) 
        colors[i] = np.trapz(filter_response, x=X, axis=1)
        
    colors = colors.transpose()
    
    return colors[0]

def log_prior(theta):
    """
    Returns log prior with constraint that 0 < theta_i < 1 and sum(theta) = 1.

    Parameters:
        theta (array): array of combinations

    Returns:
        (float or -np.inf): log probability (prior)
    """
    
    if ((np.all(theta <= 1) == True) and (np.all(theta >= 0) == True) 
        and (np.abs(np.sum(theta) - 1) < settings.MIN_EPS)):
        return 0.0
    return -np.inf

def log_likelihood(theta, y, yerr):
    """
    Returns log likelihood function for a model with only measurement error.
    Error is assumed normally distributed.

    Parameters:
        theta (array): array of combinations
        y, yerr (array): measurement and measurement error

    Returns:
        (float): log probability (likelihood)
    """
    return -0.5 * np.sum( (y - eval_model(theta))** 2 / yerr ** 2 + 
                          np.log(2*np.pi * yerr**2) )

def log_probability(theta, y, yerr):
    """
    Returns the log posterior distribution function. Composed of
    log_likelihood + log_prior from above.

    Parameters:
        theta (array): array of combinations
        y, yerr (array): measurement and measurement error

    Returns:
        (float): log probability (posterior)
    """
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, y, yerr)