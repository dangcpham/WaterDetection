from utils_mcmc import *
import emcee
import argparse
from multiprocessing import Pool

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--mode',
                    type=str, help='single or multiprocessing', required=True)
parser.add_argument('-n', '--snr',
                    type=int, help='signal to noise ratios', required=True)
args = parser.parse_args()
assert args.mode in ['single', 'multiprocessing']

# generate random true combinations
true_thetas = utils.uniform_unity(size=(settings.MCMC_N_REALIZATIONS,
                                        len(settings.labels) ) )
# important components (we only really care about water, cloud, snow)
selected_idx = [settings.labels.index(component) 
                for component in settings.CLASSIFYING_COMPONENTS]

################################ IC AND SETUP ##################################

snr = int(args.snr)

# get gaussian scatter from SNR
gaussian_scatter = 1/snr

# holds yobs and yerr for each random realization
# shape is (n_realizations, n_filters, 2)
# yobs = all_y[i,:,0]. yerr = all_y[i,:,1]
all_y = np.zeros((settings.MCMC_N_REALIZATIONS, 
                len(good_filters_values), 2) )
all_chains = []

################################ MCMC SAMPLING #################################

for i, true_theta in enumerate(true_thetas):
    
    print(f"SNR {snr}: {i+1}/{settings.MCMC_N_REALIZATIONS}")
    
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
    
    
    
    
    
    
    
    
    
    
