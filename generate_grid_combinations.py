import itertools
import numpy as np
import settings
import pickle

# generate combinations of surfaces that sum to unity
# takes some time to run

# how many surfaces
n_surfaces = len(settings.surface_paths) + 1

# how many steps for each component
composition_steps = np.arange(0, 105, settings.DELTA_STEP)
# create all possible combinations (note: can be greater than 100 here)
surface_combinations_iter = itertools.product(composition_steps, repeat=n_surfaces)
# convert to numpy array
surface_combinations = np.array(list(surface_combinations_iter))

# get combinations that sums to 100, and renormalize to 1
# this is what we want
unity_surface_combinations = surface_combinations[
    np.where(np.sum(surface_combinations, axis = 1) == 100)[0]]/100

# save data so we don't have to do this again
pickle.dump([settings.labels, unity_surface_combinations], 
        open(f"{settings.OUTPUT_DIR}/surface_combinations.pkl", "wb"))