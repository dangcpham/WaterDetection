import numpy as np
from scipy.interpolate import interp1d

def rayleigh_function(X, star_spectrum, p):
    """
        Returns the additive term for Rayleigh scattering.
        This version takes in wavelength and returns the transmission.
        Taken from Allen's Astrophysical Quantities (2000).

        Parameter:
            X (array): 1-D array of wavelength where Rayleigh scattering
                is calculated.
            star_spectrum (array): spectrum of the star, must have same
                length as X.
            p (float, optional): Pressure (ratio). One for surface.
         Returns:
            (array): Rayleigh scattering (same unit as star_spectrum)
    """
    assert len(X.shape) == 1
    assert X.shape == star_spectrum.shape
    
    rayleigh_tau  = 0.008569 * X**(-4) *(1 + 0.0113 * X**(-2) + 0.00013 * X**(-4))
    rayleigh_tau *= p
    
    rayleigh_scattering = 1 - np.exp(-rayleigh_tau)
    rayleigh = rayleigh_scattering * star_spectrum
    
    return rayleigh


def get_surface_data(paths, X):
    """
        Read csv and ascii files, given the paths.
        Then, interpolate and evaluate at wavelength X.
        
        Note: if file ends in ".csv", use csv reader,
            ends in ".asc", use ascii reader,
            else, throw error.

        Parameter:
            paths (str): list of path to the files
            X (array): wavelength to evaluate
        
        Returns:
            (array): data at X

    """
    surface_data  = np.zeros((len(paths), len(X)))
    
    for i, component_path in enumerate(paths):

        if component_path.endswith('.csv'):
            wavelength, albedo = np.genfromtxt(component_path, dtype=None, 
                                               delimiter=',',
                                               unpack=True)
        elif component_path.endswith('.asc'):
            wavelength, albedo, _= np.genfromtxt(component_path, dtype=None, 
                                                  skip_header=1, filling_values=np.NaN, 
                                                  invalid_raise=False, unpack=True)
        else:
            raise ValueError('supported file formats: csv and asc')
            
        surface_interp = interp1d(wavelength, albedo)
        surface_data[i] = surface_interp(X)
        
    return surface_data

def make_all_spectra(X, combinations, cloud_index,
                     surfaces_albedo, cloud_albedo,
                     transmission_surface, transmission_6km,
                     p_6km,
                     star_spectrum):
    """
        Create spectra at given wavelengths, given combinations
        of components.
        
        Parameters:
            X (array): the wavelengths where flux are eval.
            combinations (array): array of combinations of
                components.
            cloud_index (int): which component in combinations is
                the cloud
            surfaces_albedo (array): array of albedo components 
                at surface
            cloud_alebdo (array): array of 6 km cloud albedo
            transmission_surface (array): transmission at surface
                (note: already multiplied by star spectrum)
            transmission_6km (array): transmission at 6km
                (note: already multiplied by star spectrum)
            p_6km (float): pressure at 6 km
            star_spectrum (array): spectrum of the star
        
        Returns
            (array): array of flux. shape is (ncombinations, X.size)
            
    """
    # separate combinations into surface and cloud
    surface_combinations = np.delete(combinations, cloud_index, axis=1)
    cloud_composition =  combinations[:,cloud_index]
    # reshape so we can do matrix multiplication later
    cloud_composition = cloud_composition.reshape(cloud_composition.shape[0], 1)
    
    # surface flux
    weighted_surface_albedos = surface_combinations.dot(surfaces_albedo)
    rayleigh_surface = rayleigh_function(X, star_spectrum, p=1.)
    flux_surface = transmission_surface*weighted_surface_albedos + rayleigh_surface
    
    # cloud only at 6km
    weighted_6km = cloud_composition.dot(cloud_albedo)
    rayleigh_6km = rayleigh_function(X, star_spectrum, p=p_6km)
    flux_cloud = transmission_6km*weighted_6km + rayleigh_6km
    
    # total flux
    flux = flux_surface + flux_cloud

    return flux

def filter_func(initial, x, step):
    """
        Create simple filters with size "step", starting
        at "initial" over the wavelength "x".

        This filter is 1 between [initial, initial+step],
        0 otherwise.

        Parameters
            initial (float): initial wavelength
            x (array): list of wavelengths
            step (float): width of the filter
        
        Returns
            (array): filter behavior over x
    """
    
    output = np.zeros(len(x))
    inrange = (x >= initial) & (x < initial + step)
    
    output[inrange] = 1
    output[~inrange] = 0

    return output