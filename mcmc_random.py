from utils_mcmc import *
import emcee
import argparse
import pickle
from multiprocessing import Pool

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--mode',
                    type=str, help='single or multiprocessing',
                    required=True)
parser.add_argument('-r', '--restart',
                    type=bool, nargs='?', const=False,
                    help='restart using existing run?')
args = parser.parse_args()
assert args.mode in ['single', 'multiprocessing']
restart = args.restart

print(f'Mode: {args.mode}')
print(f'Restarting from existing runs: {restart}')

# generate random true combinations

if restart:
    try:
        true_thetas = pickle.load(
            open(f'{settings.OUTPUT_DIR}/MCMC_true_thetas.pkl', 'rb'))
    except OSError:
        true_thetas = utils.uniform_unity(size=(settings.MCMC_N_REALIZATIONS,
                                            len(settings.labels) ) )
        pickle.dump(true_thetas,
            open(f'{settings.OUTPUT_DIR}/MCMC_true_thetas.pkl', 'wb'))
else:
    true_thetas = utils.uniform_unity(size=(settings.MCMC_N_REALIZATIONS,
                                            len(settings.labels) ) )
    pickle.dump(true_thetas,
        open(f'{settings.OUTPUT_DIR}/MCMC_true_thetas.pkl', 'wb'))
    
# important components (we only really care about water, cloud, snow)
selected_idx = [settings.labels.index(component) 
                for component in settings.CLASSIFYING_COMPONENTS]

for snr in settings.MCMC_SNRS:

################################ IC AND SETUP ##################################

    # get gaussian scatter from SNR
    gaussian_scatter = 1/snr

    # all_y holds yobs and yerr for each random realization
    # shape is (n_realizations, n_filters, 2)
    # yobs = all_y[i,:,0]. yerr = all_y[i,:,1]
    # all_chains holds all sampled MCMC chains
    if restart:
        try:
            # try to load existing runs
            _,all_chains = pickle.load(
                open(f'{settings.OUTPUT_DIR}/MCMC_SNR_{snr}_true_and_chains.pkl', 'wb'))
            all_y = pickle.load(
                open(f'{settings.OUTPUT_DIR}/MCMC_SNR_{snr}_ydata.pkl', 'wb'))
        except OSError:
            # if the existing runs don't exist, start from new
            all_y = np.zeros((settings.MCMC_N_REALIZATIONS, 
                        len(good_filters_values), 2) )
            all_chains = []
    else:
        all_y = np.zeros((settings.MCMC_N_REALIZATIONS, 
                        len(good_filters_values), 2) )
        all_chains = []

    # the starting position (only for restarting)
    if restart: 
        starting_idx = len(all_chains)
    else:
        starting_idx = 0
        
################################ MCMC SAMPLING #################################

    for i, true_theta in enumerate(true_thetas[starting_idx:]):
        
        print(f"SNR {snr}: {i+starting_idx+1}/{settings.MCMC_N_REALIZATIONS}")
        
        # generate noiseless data
        ytrue = eval_model(true_theta)
        
        # add noise to data
        yerr = ytrue*gaussian_scatter*np.random.standard_normal(ytrue.shape)
        yobs = ytrue + yerr
        
        # save y's
        all_y[i,:,0] = yobs
        all_y[i,:,1] = yerr
        
        # draw random positions
        pos = utils.uniform_unity(size=(settings.MCMC_N_WALKERS,
                                        len(settings.labels)))
        nwalkers, ndim = pos.shape
        
        if args.mode == 'multiprocessing':
            with Pool() as pool:
                # start the sampler
                sampler = emcee.EnsembleSampler(
                        nwalkers, ndim, log_probability, args=(yobs, yerr),
                        pool=pool
                )        
            
                # start sampling
                sampler.run_mcmc(pos, settings.MCMC_CHAINS_LEN, progress=True,
                                skip_initial_state_check=True)
        else:
            # start the sampler
            sampler = emcee.EnsembleSampler(
                    nwalkers, ndim, log_probability, args=(yobs, yerr)
            )        
        
            # start sampling
            sampler.run_mcmc(pos, settings.MCMC_CHAINS_LEN, progress=True,
                            skip_initial_state_check=True)
            
        # get burn-in and thinning from ACT
        tau = sampler.get_autocorr_time()[selected_idx]
        burnin = int(2*np.max(tau))
        thin = int(0.5*np.min(tau))
        samples = sampler.get_chain(discard=burnin, flat=True, thin=thin)
        
        # save to array
        all_chains.append(samples)
        
        # save data
        pickle.dump((true_thetas, all_chains), 
            open(f'{settings.OUTPUT_DIR}/MCMC_SNR_{snr}_true_and_chains.pkl', 'wb'))
        pickle.dump(all_y, 
            open(f'{settings.OUTPUT_DIR}/MCMC_SNR_{snr}_ydata.pkl', 'wb'))
        
    
    
    
    
    
    
    
    
    
