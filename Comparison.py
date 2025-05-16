import os
os.environ["OMP_NUM_THREADS"] = "5" # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = '1'

import pickle as pk
import pandas as pd
import random
import numpy as np
import sys
from datetime import datetime
from multiprocessing import Pool
import warnings
import glob

#SEMISUPERVISED MODELS
sys.path.insert(1, '/home/aolivares/sslearn')
from sslearn.wrapper import TriTrainingRegressor, CoReg

#BASE ESTIMATORS
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.tree import  DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge

#METRICS
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

#OTHER UTILS
from sklearn.model_selection import GridSearchCV
from utils.SSKFold import SSKFold, SSStratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, KFold, RepeatedStratifiedKFold



def crossVal(dataset_name, general_model, percentage_label, random_state, tune, repetitions, folds): 

    #read data 
    X = pd.read_csv(f'{dataset_name}/X.csv')
    y = pd.read_csv(f'{dataset_name}/y.csv')

    #for controlling that train and test splits are stratify by "y" quartile
    y_categorical = np.where(y <= np.percentile(y, 25), 'Q1', 
                             np.where(y <= np.percentile(y, 50), 'Q2', 
                                      np.where(y <= np.percentile(y, 75), 'Q3', 'Q4')))

    rcv = RepeatedStratifiedKFold(n_splits=folds, n_repeats=repetitions, random_state=random_state)

    args = []
    for i, (train_index, test_index) in enumerate(rcv.split(X, y_categorical)):
        args.append((i, train_index, test_index, percentage_label, dataset_name, general_model, tune))

    with Pool(1) as pool: 
        results = pool.starmap(job, args, chunksize=1)

    predictions = [x[0] for x in results]
    scores = [x[1] for x in results]
    tuned_params = [x[2] for x in results]

    return predictions, scores, tuned_params 



def job(i, train_index, test_index, percentage_label, dataset_name, general_model, tune): 
    
    random_state = 1234
    
    #read data 
    X = pd.read_csv(f'{dataset_name}/X_norm.csv')
    y = pd.read_csv(f'{dataset_name}/y.csv')

    #dictionaries to save results
    scores_dict = dict()
    predictions_dict = dict()
    tuned_params = dict()

    #split data
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    #save real "y" values (to potentially calculate new metrics)
    predictions_dict['y_test'] = y_test
    
    #unlabel part of the training set
    label_index, unlabel_index, _, _ = train_test_split(X_train.index, y_train.index, train_size=percentage_label, 
                                                        random_state=random_state)
    
    X_train_labeled, X_train_unlabeled = X_train.loc[label_index], X_train.loc[unlabel_index]
    y_train_labeled = y_train.loc[label_index]

    #Building training set with the labeled and unlabeled instances
    X_train_final = pd.concat([X_train_labeled, 
                               X_train_unlabeled], axis=0).reset_index(drop=True)
    y_col_name = y_train_labeled.columns[0]
    y_train_final = pd.concat([y_train_labeled, 
                               pd.DataFrame({y_col_name: [np.nan]*len(X_train_unlabeled)})], axis=0).reset_index(drop=True)

    if 'TriTrainingRegressor' == general_model: 

        base_regressors = {'rf': RandomForestRegressor(), 
                           'ab': AdaBoostRegressor(), 
                           'dt': DecisionTreeRegressor(), 
                           'r': Ridge(), 
                           'svm': SVR(), 
                           'knn': KNeighborsRegressor()
                          }
        for key in base_regressors: 
            print(datetime.now(), f'--> TriTrainingRegressor + {key} (split {i} dataset {dataset_name})')
            regressor = base_regressors[key]
            tritr = TriTrainingRegressor(base_estimator=regressor)

            if tune: 
                grid = {'y_tol_per': [0.000001, 0.0001, 0.001, 0.01, 0.1]}
                inner_cv = SSKFold(n_splits=3, shuffle=True, random_state=random_state)
                search = GridSearchCV(tritr, grid, cv=inner_cv)
                result = search.fit(X_train_final, y_train_final)
                best_model = result.best_estimator_
                tuned_params[f'tritr_{key}'] = result
                prediction_tritr = best_model.predict(X_test)
            else: 
                tritr.fit(X_train_final, y_train_final)
                prediction_tritr = tritr.predict(X_test)
                
            #scores
            predictions_dict[f'prediction_tritr_{key}'] = prediction_tritr
            scores_dict[f'mae_tritr_{key}'] = mean_absolute_error(y_test, prediction_tritr)
            scores_dict[f'mse_tritr_{key}'] = mean_squared_error(y_test, prediction_tritr)
            scores_dict[f'r2_tritr_{key}'] = r2_score(y_test, prediction_tritr)

    if 'CoRegression' == general_model: 
        
        print(datetime.now(), f'--> CoRegression (split {i} dataset {dataset_name})')

        cor = CoReg(max_iterations=100, pool_size=100)

        if tune: 
            grid = [{'p1': [2], 'p2': [3, 4, 5]},
                    {'p1': [3], 'p2': [4, 5]}, 
                    {'p1': [4], 'p2': [5]}]
            inner_cv = SSKFold(n_splits=3, shuffle=True, random_state=random_state)
            search = GridSearchCV(cor, grid, cv=inner_cv, error_score='raise')
            result = search.fit(X_train_final, y_train_final)
            best_model = result.best_estimator_
            tuned_params['cor'] = result
            prediction_cor = best_model.predict(X_test)
        else: 
            cor.fit(X_train_final, y_train_final)
            prediction_cor = cor.predict(X_test)
            
        #scores
        predictions_dict['prediction_cor'] = prediction_cor
        scores_dict['mae_cor'] = mean_absolute_error(y_test, prediction_cor)
        scores_dict['mse_cor'] = mean_squared_error(y_test, prediction_cor)
        scores_dict['r2_cor'] = r2_score(y_test, prediction_cor)

    return predictions_dict, scores_dict, tuned_params
    


if __name__=="__main__": 

    #experimental settings
    datasets = glob.glob("data/*")
    datasets.sort()

    datasets = [ 
        'data/2dplanes',
        'data/abalone',
        'data/add10',
        'data/ail',
        'data/airfoil_self_noise',
        'data/ana',
        'data/auto_mpg',
        'data/bank32',
        'data/bank8',
        'data/bas',
        'data/bike_sharing',
        'data/boston',
        'data/ca',
        'data/cal',
        'data/casp',
        'data/combined_cycle_power_plant',
        'data/communities_and_crime',
        'data/concrete_compressive_strength',
        'data/cpu_small',
        'data/dee',
        'data/deltaail',
        'data/deltaelv',
        'data/diamond',
        'data/ele1',
        'data/ele2',
        'data/electrical_grid_stability_simulated_data',
        'data/elv',
        'data/energy_efficiency_cooling_load',
        'data/energy_efficiency_heating_load',
        'data/facebook_metrics_comments',
        'data/facebook_metrics_likes',
        'data/facebook_metrics_shares',
        'data/facebook_metrics_total_interactions',
        'data/fat',
        'data/forest_fires',
        'data/fried',
        'data/house16',
        'data/house8',
        'data/kine32',
        'data/kine8',
        'data/laser',
        'data/liver_disorders',
        'data/machinecpu',
        'data/metro_interstate traffic_volume',
        'data/mor',
        'data/mv',
        'data/parkinson_telemonitoring_motor_updrs',
        'data/parkinson_telemonitoring_total_updrs',
        'data/pla',
        'data/pole',
        'data/puma32',
        'data/puma8',
        'data/qua',
        'data/real_state_valuation',
        'data/solar_flare_common_flares',
        'data/solar_flare_moderate_flares',
        'data/solar_flare_severe_flares',
        'data/sto',
        'data/strikes',
        'data/student_performance_math',
        'data/student_performance_portuguese',
        'data/superconductivity_data',
        'data/tre',
        'data/wan',
        'data/wine_quality_red',
        'data/wine_quality_white',
        'data/wiz',
        'data/yh'
    ]

    models = [ 'TriTrainingRegressor', 'CoRegression']
    tune = True    
    percentage_label = [0.1, 0.2, 0.3, 0.4, 0.5]
    repetitions = 5
    folds = 2

    for model in models: 
        for dataset in datasets: 
            dataset_name = dataset[dataset.find('/')+1:]
            for per in percentage_label: 
                print(datetime.now(), 'DATASET:', dataset, 'MODEL:', model, 'N labeled:', per)
                predictions, scores, tuned_params = crossVal(dataset, model, per, random_state=1234, 
                                                             tune=tune, repetitions=repetitions, folds=folds)
                #save results
                with open(f'results/predictions_comparison_{model}_{dataset_name}_{str(per)}_per.pk', 'wb') as file_predictions: 
                    pk.dump(predictions, file_predictions)
                with open(f'results/scores_comparison_{model}_{dataset_name}_{str(per)}_per.pk', 'wb') as file_scores: 
                    pk.dump(scores, file_scores)
                with open(f'results/tuned_params_comparison_{model}_{dataset_name}_{str(per)}_per.pk', 'wb') as file_tuned_params: 
                    pk.dump(tuned_params, file_tuned_params)
                        
        
        