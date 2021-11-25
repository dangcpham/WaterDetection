import numpy as np
from scipy.interpolate import interp1d

def reshape_to_2D(y):
    """
        Reshape an array with shape (n,) to (n,1).
        
        Parameters:
            y (array): the array with shape (n,)
        
        Returns:
            (array): array with shape (n,1)
    """
    
    return y.reshape(y.shape[0], 1)

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
#     assert len(X.shape) == 1
#     assert X.shape == star_spectrum.shape
    
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
    cloud_composition =  reshape_to_2D(combinations[:,cloud_index])
    
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

def uniform_unity(a=0., b=1., size=1):
    """
        Random variable drawing from uniform distribution such that
        the sum of weights (in n-dimensions) is unity. That is,
        we are drawing random variables on the (n-1) simplex.
    
        Algorithm from whuber's answer to
        https://stats.stackexchange.com/questions/14059/generate-uniformly-distributed-weights-that-sum-to-unity
        
        Parameters:
            a, b (float): range of the uniform distribution, inclusive.
                Optional. Default is (a,b) = (0, 1)
            size (tuple): (numbero of samples, number of dimensions).
                Optional. Default is (1, 1) (one sample, one dimension).
            
        Returns:
            (array): random variables with shape (size).
    """
    
    # x ~ U(a,b)
    x = np.random.uniform(low=a, high=b, size=size)
    # y = -log(x), so that each y_i has Gamma(1) distribution
    y = -np.log(x)
    # y_1 + ... y_n = sum(y)
    y_sum = np.sum(y, axis=1)
    # vector reshaping things so we can divide next
    y_sum = reshape_to_2D(y_sum)
    # normalize y/sum(y) => w is Dirichlet(1,...,1) = Uniform
    # with sum(w) = 1
    w = y/y_sum
    
    return w

def get_earth_composition(cloud_percent):
    """
        Give earth surface composition percentage.
        Land percentage given from Kaltenegger (2007) 
    """
    cloud_composition = cloud_percent

    seawater_composition = 0.70*(100-cloud_composition)
    land_composition = 0.30*(100-cloud_composition)
    
    basalt_composition = 0.18*land_composition
    snow_composition = 0.15*land_composition
    sand_composition = 0.07*land_composition
    tree_composition = 0.6*land_composition

    earth_composition = {'cloud': cloud_composition,
                         'seawater': seawater_composition, 
                         'basalt': basalt_composition, 
                         'snow': snow_composition, 
                         'veg': tree_composition, 
                         'sand': sand_composition
                        }
    
    return earth_composition