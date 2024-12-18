import numpy as np
import settings
import pickle
from utils import filter_func

X = np.linspace(settings.WAVELENGTH_MIN, settings.WAVELENGTH_MAX, 
                settings.WAVELENGTH_RESOLUTION)
all_spectra = pickle.load(open(f"{settings.OUTPUT_DIR}/all_spectra.pkl", "rb"))

# how many filters can we fit within the range
N_FILTERS = int(np.floor((settings.FILTER_MAX - settings.FILTER_MIN) / 
                            settings.FILTER_SIZE))

# allocate array to hold colors
colors = np.zeros((N_FILTERS, len(all_spectra)))

for i in range(N_FILTERS):
    # get filter response locations
    filter_pos = settings.FILTER_MIN + settings.FILTER_SIZE*(i)
    # multiply spectra by filter response
    filter_response = all_spectra * filter_func(filter_pos, X, 
                                                step=settings.FILTER_SIZE)
    # integrate filters (trapezoidal rule works well enough at high res.)
    colors[i] = np.trapz(filter_response, x=X, axis=1)

colors = colors.transpose()

# save colors data
filter_names = [f"f{i}" for i in range(N_FILTERS)]
pickle.dump((filter_names, colors), 
            open(f"{settings.OUTPUT_DIR}/colors_f1.pkl", "wb"))

# save filters data
filter_info = {f"f{i}":settings.FILTER_MIN + settings.FILTER_SIZE*(i) 
               for i in range(N_FILTERS)}
pickle.dump(filter_info, open(f"{settings.OUTPUT_DIR}/filters_f1.pkl", "wb"))