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



def crossVal(dataset_name, percentage_label, random_state, repetitions, folds): 

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
        args.append((i, train_index, test_index, percentage_label, dataset_name))

    with Pool(1) as pool: 
        results = pool.starmap(job, args, chunksize=1)

    predictions = [x[0] for x in results]
    scores = [x[1] for x in results]

    return predictions, scores



def job(i, train_index, test_index, percentage_label, dataset_name): 
    
    random_state = 1234
    
    #read data 
    X = pd.read_csv(f'{dataset_name}/X_norm.csv')
    y = pd.read_csv(f'{dataset_name}/y.csv')

    #dictionaries to save results
    scores_dict = dict()
    predictions_dict = dict()

    #split data
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    #save real "y" values (to potentially calculate new metrics)
    predictions_dict['y_test'] = y_test
    
    #unlabel part of the training set
    label_index, unlabel_index, _, _ = train_test_split(X_train.index, y_train.index, train_size=percentage_label, 
                                                        random_state=random_state)
    
    X_train_labeled, X_train_unlabeled = X_train.loc[label_index], X_train.loc[unlabel_index]
    y_train_labeled = y_train.loc[label_index].to_numpy().flatten()

    models = {'rf': RandomForestRegressor(), 
              'ab': AdaBoostRegressor(),
              'dt': DecisionTreeRegressor(), 
              'r': Ridge(), 
              'svm': SVR(), 
              'knn': KNeighborsRegressor()
             }

    for key, model in models.items(): 
        
        print(datetime.now(), f'--> {key} (split {i} dataset {dataset_name})')
        model.fit(X_train_labeled, y_train_labeled)
        prediction = model.predict(X_test)

        #scores
        predictions_dict[f'prediction_{key}'] = prediction
        scores_dict[f'mae_{key}'] = mean_absolute_error(y_test, prediction)
        scores_dict[f'mse_{key}'] = mean_squared_error(y_test, prediction)
        scores_dict[f'r2_{key}'] = r2_score(y_test, prediction)

    return predictions_dict, scores_dict
    


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
 
    percentage_label = [0.1, 0.2, 0.3, 0.4, 0.5]
    repetitions = 5
    folds = 2

    for dataset in datasets: 
        dataset_name = dataset[dataset.find('/')+1:]
        for per in percentage_label: 
            print(datetime.now(), 'DATASET:', dataset, 'N labeled:', per)
            predictions, scores = crossVal(dataset, per, random_state=1234, repetitions=repetitions, folds=folds)
            
            #save results
            with open(f'results/predictions_comparison_Supervised_{dataset_name}_{str(per)}_per.pk', 'wb') as file_predictions: 
                pk.dump(predictions, file_predictions)
            with open(f'results/scores_comparison_Supervised_{dataset_name}_{str(per)}_per.pk', 'wb') as file_scores: 
                pk.dump(scores, file_scores)

                        
        
        