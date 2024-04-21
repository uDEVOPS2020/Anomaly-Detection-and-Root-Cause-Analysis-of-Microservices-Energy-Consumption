#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: li
"""

import pandas as pd
import numpy as np
import networkx as nx
import csv
import os
import shutil
from sklearn.cluster import Birch
from sklearn import preprocessing


# SYSTEM = 'sockshop'
SYSTEM = 'unicloud'
# Define the main directory
DATA_FOLDER = f'C:\scul\Thesis\Data Analysis\{SYSTEM}-data'
smoothing_window = 12
TOP_K = 3

# Anomaly Detection
def birch_ad_with_smoothing(df, threshold):    
    anomalies = []
    df = df.filter(like='_energy')    
    # Assuming 'df' contains the filtered columns with '_energy'
    # Rename columns to remove '_energy'
    new_column_names = {col: col.replace('_energy', '') for col in df.columns}
    df = df.rename(columns=new_column_names)
    
    for svc, power in df.items():
        # No anomaly detection in db
        if svc != 'time' and 'Unnamed' not in svc and 'rabbitmq' not in svc and 'db' not in svc:
            power = power.rolling(window=smoothing_window, min_periods=1).mean()
            x = np.array(power)
            x = np.where(np.isnan(x), 0, x)
            normalized_x = preprocessing.normalize([x])

            X = normalized_x.reshape(-1,1)

            brc = Birch(branching_factor=50, n_clusters=None, threshold=threshold, compute_labels=True)
            brc.fit(X)
            brc.predict(X)

            labels = brc.labels_
            n_clusters = np.unique(labels).size
            if n_clusters > 1:
                anomalies.append(svc)
    return anomalies

def energy_invocations(file_path):
        
    df = pd.read_csv(file_path)

    return df

def attributed_graph(service_links_df):
    # build the attributed graph 
    # input: prefix of the file
    # output: attributed graph

    DG = nx.DiGraph()    
    for index, row in service_links_df.iterrows():
        source = row['source']
        destination = row['destination']
        if 'rabbitmq' not in source and 'rabbitmq' not in destination and 'db' not in destination and 'db' not in source:
            DG.add_edge(source, destination)
                
    return DG 

def svc_personalization(svc, anomaly_graph, baseline_df, df):
    ctn_cols = [f'{svc}_cpu', f'{svc}_memory']
    max_corr = 0.01
    for col in ctn_cols:
        temp = abs(baseline_df[svc].corr(df[col]))     
        if temp > max_corr:
            max_corr = temp


    edges_weight_avg = 0.0
    num = 0
    for u, v, data in anomaly_graph.in_edges(svc, data=True):
        num = num + 1
        edges_weight_avg = edges_weight_avg + data['weight']

    edges_weight_avg  = edges_weight_avg / num

    personalization = edges_weight_avg * max_corr

    return personalization

def anomaly_subgraph(DG, anomalies, df, alpha, metric):
    # Get the anomalous subgraph and rank the anomalous services
    # input: 
    #   DG: attributed graph
    #   anomlies: anoamlous service invocations
    #   latency_df: service invocations from data collection
    #   agg_latency_dff: aggregated service invocation
    #   faults_name: prefix of csv file
    #   alpha: weight of the anomalous edge
    # output:
    #   anomalous scores 
    # Get reported anomalous nodes
    edges = []
    nodes = []
    baseline_df = pd.DataFrame()
    for anomaly in anomalies:
        edge = anomaly.split('_')
        edges.append(tuple(edge))
        svc = edge[1]
        svc_metric = f'{svc}_{metric}'
        nodes.append(svc)
        baseline_df[svc] = df[svc_metric]

    nodes = set(nodes)

    personalization = {}
    for node in DG.nodes():
        if node in nodes:
            personalization[node] = 0

    # Get the subgraph of anomaly
    anomaly_graph = nx.DiGraph()
    # print(f'Nodes = {nodes}')
    for node in nodes:
        for u, v, data in DG.in_edges(node, data=True):
            
            edge = (u,v)
            # print(f'Incoming Edge = {edge}')
            if edge in edges:
                data = alpha
            else:
                data = baseline_df[v].corr(df[f'{u}_{metric}'])

            data = round(data, 3)
            anomaly_graph.add_edge(u,v, weight=data)

       # Set personalization with container resource usage
        for u, v, data in DG.out_edges(node, data=True):
            edge = (u,v)
            # print(f'Outgoing Edge = {edge}')
            if edge in edges:
                data = alpha
            else:
                data = baseline_df[u].corr(df[f'{v}_{metric}'])
            data = round(data, 3)
            anomaly_graph.add_edge(u,v, weight=data)


    for node in nodes:
        max_corr = svc_personalization(node, anomaly_graph, baseline_df, df)
        personalization[node] = max_corr / anomaly_graph.degree(node)

    anomaly_graph = anomaly_graph.reverse(copy=True)

    edges = list(anomaly_graph.edges(data=True))

    anomaly_score = nx.pagerank(anomaly_graph, alpha=0.85, personalization=personalization, max_iter=10000)

    anomaly_score = sorted(anomaly_score.items(), key=lambda x: x[1], reverse=True)

    return anomaly_score

def print_results(results, model, path):   

    # Extracting node names
    nodes_list = [node[0] for node in results[:3]]
    # Writing nodes_list to a CSV file
    csv_filename = f'{path}\{model}_results.csv'

    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Root Cause'])
        writer.writerows([[node] for node in nodes_list])
        
if __name__ == '__main__':
    
    # Tuning parameters
    alpha = 0.55  
    ad_threshold = 0.045 
        
    fault_type = 'energy'
    
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
                            trial_path = os.path.join(service_path, stress, user, scenario, time_window, trial)       
                            data_file_path = os.path.join(service_path, stress, user, scenario, time_window, trial, 'data.csv')
                            
                            df = energy_invocations(data_file_path)
                            
                            threshold = ad_threshold   
                            
                            anomalies = birch_ad_with_smoothing(df, threshold)
                            # print(f'Anomalous services {anomalies}') 
                            # anomalies_by_metric = {}
                            # for anomaly in anomalies:
                            #     svc = anomaly.split('_')[0]
                            #     if 'memory_rss' in anomaly or 'memory_cache' in anomaly:                                    
                            #         # metric = f"{anomaly.split('_')[1]}_{anomaly.split('_')[2]}"                                    
                            #         continue
                            #     else:
                            #         metric = anomaly.split('_')[1]
                            #     if not metric in anomalies_by_metric:
                            #         anomalies_by_metric[metric] = [svc]
                            #     else:
                            #         anomalies_by_metric[metric] = np.concatenate((anomalies_by_metric[metric], svc), axis = None)
                            service_links_df = pd.read_csv(f'C:\scul\Thesis\Data Analysis\RCA\MicroRCA\{SYSTEM}.csv')
                            
                            anomaly_score_by_metric = {}
                            # Extract unique source-destination pairs from the CSV
                            # for metric in anomalies_by_metric:   
                            #     anomalies = anomalies_by_metric[metric]                             
                            source_destination_pairs = service_links_df[service_links_df['source'].isin(anomalies) & service_links_df['destination'].isin(anomalies)][['source', 'destination']]
                            if source_destination_pairs.empty:
                                continue
                            anomalous_pairs = source_destination_pairs.apply(lambda x: '_'.join(x), axis=1).tolist()

                                # print(f'Anomalous pairs {anomalous_pairs}')
                                # construct attributed graph
                            DG = attributed_graph(service_links_df)
                            anomaly_score = anomaly_subgraph(DG, anomalous_pairs, df, alpha, 'energy')
                            # anomaly_score_by_metric[metric] = anomaly_score

                            # merged_and_sorted_list = []

                            # for values in anomaly_score_by_metric.values():
                            #     merged_and_sorted_list.extend(values)

                            # merged_and_sorted_list = sorted(merged_and_sorted_list, key=lambda x: x[1], reverse=True)

                            print_results(anomaly_score, 'MicroRCA', trial_path)

