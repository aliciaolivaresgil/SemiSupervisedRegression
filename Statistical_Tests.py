import scipy.stats as stats
import scikit_posthocs as sp
import cv2

import csv
import sys
import baycomp as bc

import pickle as pk
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import warnings 
import math

from utils.baycomp_plotting import tern 

names_map = {
    'tritr_rf': 'TriTR+RF', 
    'tritr_ab': 'TriTR+AB', 
    'tritr_dt': 'TriTR+DT', 
    'tritr_r': 'TriTR+Ridge', 
    'tritr_svm': 'TriTR+SVM', 
    'tritr_knn': 'TriTR+K-NN', 
    'cor': 'COREG', 
    'rf': 'RF', 
    'ab': 'AB', 
    'dt': 'DT', 
    'r': 'Ridge', 
    'svm': 'SVM', 
    'knn': 'K-NN'
}

def bayesian(model1, model2, data1, data2, metric, rope=0.05, per_label=0.5): 

    data1 = np.array(data1)
    data2 = np.array(data2)
    posterior = bc.HierarchicalTest(data1, data2, rope=rope)
    with open(f'results_baycomp/bayesian_posteriors_rope={rope}_{model1}_{model2}_{metric}_{per_label}.pk', 'wb') as f: 
        pk.dump(posterior, f)

def generatePlots(model1, model2, metric, rope, per_label): 
    posterior = pk.load(open(f'results_baycomp/bayesian_posteriors_rope={rope}_{model1}_{model2}_{metric}_{per_label}.pk', 'rb'))
    fig = tern(posterior, l_tag=names_map[model1], r_tag=names_map[model2])
    plt.savefig(f'figs/bayesian_rope={rope}_{model1}_{model2}_{metric}_{per_label}.pdf')

    matplotlib.pyplot.close()

if __name__=="__main__": 

    datasets = ['2dplanes', 'abalone', 'add10', 'airfoil_self_noise', 'ana', 'auto_mpg', 'bank32', 'bank8', 'bas', 'bike_sharing','boston',
                'ca', 'cal', 'casp', 'combined_cycle_power_plant', 'communities_and_crime', 'concrete_compressive_strength', 'cpu_small', 
                'dee', 'diamond','ele1', 'ele2', 'electrical_grid_stability_simulated_data','energy_efficiency_cooling_load', 
                'energy_efficiency_heating_load', 'fat', 'fried', 'house16', 'house8', 'kine32', 'kine8', 'laser', 'liver_disorders', 
                'machinecpu', 'metro_interstate traffic_volume', 'mor', 'mv', 'parkinson_telemonitoring_motor_updrs', 
                'parkinson_telemonitoring_total_updrs', 'pla', 'pole', 'puma32', 'puma8', 'qua', 'real_state_valuation', 'sto',
                'student_performance_math', 'student_performance_portuguese', 'superconductivity_data', 'tre', 'wan', 'wine_quality_red',
                'wine_quality_white', 'wiz', 'yh'
               ]

    base_regressors_map = {
                       'rf': [('TriTrainingRegressor', 'tritr_rf')], 
                       'ab': [('TriTrainingRegressor', 'tritr_ab')], 
                       'dt': [('TriTrainingRegressor', 'tritr_dt')], 
                       'r':  [('TriTrainingRegressor', 'tritr_r')], 
                       'svm':[('TriTrainingRegressor', 'tritr_svm')], 
                       'knn': [('TriTrainingRegressor', 'tritr_knn'), 
                              ('CoRegression', 'cor')
                              ]
                     }


    percentage_label = ['0.5', '0.4', '0.3', '0.2', '0.1']
    metrics = ['r2']

    for per in percentage_label: 
        for metric in metrics: 
            for base_regressor in base_regressors_map: 
                for general_model, model in base_regressors_map[base_regressor]: 

                    supervised_data = []
                    for dataset in datasets: 
                        scores = pk.load(open(f'results/scores_comparison_Supervised_{dataset}_{str(per)}_per.pk', 'rb'))
                        supervised_data.append([s[f'{metric}_{base_regressor}'] for s in scores])

                    semi_data = []
                    for dataset in datasets: 
                        scores = pk.load(open(f'results/scores_comparison_{general_model}_{dataset}_{str(per)}_per.pk', 'rb'))
                        semi_data.append([s[f'{metric}_{model}'] for s in scores])

                    bayesian(base_regressor, model, supervised_data, semi_data, metric, rope=0.05, per_label=per)
                    generatePlots(base_regressor, model, metric, rope=0.05, per_label=per)