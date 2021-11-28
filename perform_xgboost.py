import numpy as np
import pickle
import pandas as pd
import settings
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import balanced_accuracy_score
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--mode',
                    type=str, help='train using "all" or "good" filters?',
                    required=True)
args = parser.parse_args()
assert args.mode in ['all', 'good']
mode = args.mode

# get generated data
filter_names, colors = pickle.load(open(f"{settings.OUTPUT_DIR}/colors_f1.pkl", "rb"))

if mode == 'good':
    # load good filters
    good_filters_info = pickle.load(
        open(f"{settings.OUTPUT_DIR}/good_filters_info.pkl", "rb"))
    good_filters_idx = np.where(np.isin(
        filter_names, list(good_filters_info.keys()) ))[0]
    colors = colors[:,good_filters_idx]

# load all component combinations that sum to unity
component_names, unity_surface_combinations = pickle.load(
    open(f"{settings.OUTPUT_DIR}/surface_combinations.pkl", "rb"))
unity_surface_combinations_df = pd.DataFrame(unity_surface_combinations, 
                                             columns=component_names)

# store all trained models
xgb_clfs = []
# storing balanced accuracy at no noise S/N
ba_score_no_noise = np.zeros(len(settings.CLASSIFYING_COMPONENTS))
# storing feature importance scores
feature_importance = np.zeros(
    (len(settings.CLASSIFYING_COMPONENTS), len(filter_names))
    )

for i, component in enumerate(settings.CLASSIFYING_COMPONENTS):
    # get training data and labels for machine learning
    X = colors
    y = (unity_surface_combinations_df[component] > 0).astype(int)
    
    X_train, X_test_no_noise, y_train, y_test= train_test_split(
                X, y, test_size=0.2, shuffle=True)
    
    # train XGBoost
    model = XGBClassifier(objective="binary:logistic", 
        tree_method = "auto",
        scale_pos_weight = len(y_train[y_train == 0])/len(y_train[y_train > 0]),
        gpu_id = 0, 
        use_label_encoder=False,
        eval_metric=balanced_accuracy_score)
    model.fit(X_train, y_train)
    print(f"{component}: Finished training XGBoost")
    
    # append to list of trained models
    xgb_clfs.append(model)
    
    # evaluate model at no noise
    y_pred_no_noise = model.predict(X_test_no_noise)
    
    # get balanced accuracy and feature importance
    ba_score_no_noise[i] = balanced_accuracy_score(y_test, y_pred_no_noise)
    feature_importance[i] = model.feature_importances_
    
    # evaluate at various snrs
    y_pred = np.zeros((len(settings.SNRS), settings.RANDOM_INITIALIZATIONS, 
                        y_test.shape[0]))
    ba_scores = np.zeros((len(settings.SNRS), settings.RANDOM_INITIALIZATIONS))
    
    for j, snr in enumerate(settings.SNRS):
        # Gaussian sigma = 1/snr
        gaussian_scatter = 1/snr
        
        for k in range(settings.RANDOM_INITIALIZATIONS):
            # add noise to testing dataset
            noise_array = ( X_test_no_noise * 
                gaussian_scatter*np.random.standard_normal(X_test_no_noise.shape) )
            X_test = X_test_no_noise + noise_array
            
            # predict
            y_pred[j, k] = model.predict(X_test)
            ba_scores[j, k] = balanced_accuracy_score(y_test, y_pred[j, k])
    
    # save data
    pickle.dump((model, y_test, settings.SNRS, y_pred_no_noise, y_pred, 
                ba_score_no_noise[i], ba_scores), 
                open(f"{settings.OUTPUT_DIR}/result_{component}_{mode}.pkl", "wb"))

pickle.dump((xgb_clfs, feature_importance), 
            open(f"{settings.OUTPUT_DIR}/models_and_features_{mode}.pkl", "wb"))