import csv
import os
from pprint import pprint
from pyrca.analyzers.epsilon_diagnosis import EpsilonDiagnosis, EpsilonDiagnosisConfig
import pandas as pd
from pyrca.analyzers.bayesian import BayesianNetwork
from pyrca.graphs.causal.pc import PC
from pyrca.analyzers.rcd import RCD
from pyrca.analyzers.random_walk import RandomWalk, RandomWalkConfig

# SYSTEM = 'sockshop'
SYSTEM = 'unicloud'
# Define the main directory
DATA_FOLDER = f'C:\scul\Thesis\Data Analysis\{SYSTEM}-data'
TRAIN_DATA_FOLDER = f'C:\scul\Thesis\Data Analysis\{SYSTEM}-data\normal'
NORMAL_DATA_PATH = os.path.join(DATA_FOLDER, "normal")
K = 3
col_to_drop = "time" 
def print_results(results, model, trial_path):   

    # Extracting node names
    nodes_list = [node[0] for node in results['root_cause_nodes']]

    # Writing nodes_list to a CSV file
    csv_filename = f'{trial_path}\{model}_results.csv'

    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Root Cause'])
        writer.writerows([[node] for node in nodes_list])
        
def run_epsilon_diagnosis(train_trial_path, test_trial_path, trial_path):
    model = EpsilonDiagnosis(config=EpsilonDiagnosis.config_class(alpha=0.05, root_cause_top_k = K))
    
    train_df = pd.read_csv(train_trial_path)
    train_df.drop(columns = col_to_drop, inplace=True)
    train_df = train_df.filter(like='_energy')    
    
    test_df = pd.read_csv(test_trial_path)
    test_df.drop(columns = col_to_drop, inplace=True)
    test_df = test_df.filter(like='_energy')    
    
    model.train(train_df)    
    results = model.find_root_causes(test_df)
    print_results(results.to_dict(), 'epsilon', trial_path)
    
def run_rcd(train_trial_path, test_trial_path, trial_path):
    model = RCD(config=RCD.config_class(start_alpha=0.05, k = K, bins= 5, gamma=5, localized=True))
    
    train_df = pd.read_csv(train_trial_path)
    train_df.drop(columns = col_to_drop, inplace=True)
    train_df = train_df.filter(like='_energy')    
    
    test_df = pd.read_csv(test_trial_path)
    test_df.drop(columns = col_to_drop, inplace=True)
    test_df = test_df.filter(like='_energy')    
    
    results = model.find_root_causes(train_df, test_df)
    print_results(results.to_dict(), 'rcd', trial_path)
    
if __name__ == '__main__':
    
    for service in os.listdir(DATA_FOLDER):
        service_path = os.path.join(DATA_FOLDER, service)
        if not os.path.isdir(service_path):
            continue
        if service == "normal":  
            continue
        stress_factors = [folder for folder in os.listdir(service_path) if os.path.isdir(os.path.join(service_path, folder))]
        for stress in stress_factors:            
            user_factors = [folder for folder in os.listdir(os.path.join(service_path, stress)) if os.path.isdir(os.path.join(service_path, stress, folder))]
            for user in user_factors:            
                scenario_factors = [folder for folder in os.listdir(os.path.join(service_path, stress, user)) if os.path.isdir(os.path.join(service_path, stress, user, folder))]
                for scenario in scenario_factors:
                    time_windows = [folder for folder in os.listdir(os.path.join(service_path, stress, user, scenario)) if os.path.isdir(os.path.join(service_path, stress, user, scenario, folder))]
                    for time_window in time_windows:
                        trials = [folder for folder in os.listdir(os.path.join(service_path, stress, user, scenario, time_window)) if os.path.isdir(os.path.join(service_path, stress, user, scenario, time_window, folder))]
                        for trial in trials:
                            trial_path =  os.path.join(service_path, stress, user, scenario, time_window, trial)       
                            test_trial_path = os.path.join(service_path, stress, user, scenario, time_window, trial, 'data.csv')
                            train_trial_path = os.path.join(NORMAL_DATA_PATH, user, scenario, time_window, trial, 'data.csv')
                            run_epsilon_diagnosis(train_trial_path, test_trial_path, trial_path)
                            run_rcd(train_trial_path, test_trial_path, trial_path)

                            
