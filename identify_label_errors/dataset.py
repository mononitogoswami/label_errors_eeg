####################################################
#  Code to read and process dataset
####################################################


import pandas as pd
import numpy as np
from collections import Counter
from .utils import Config

args = Config(config_file_path='config.yaml').parse()

def load_data():
    # Load time-series data
    data = pd.read_csv(args['data_path']) # Read data
    expert_labels = read_and_process_labels()

    return data, expert_labels
 
def read_and_process_labels():
    # Read the expert labels
    expert_labels = pd.read_excel(args['expert_labels_path'], sheet_name="Sheet1")

    # Drop all rows for which all columns are NaN
    expert_labels.dropna(axis='index', how='all', inplace=True)

    # The same values are not repeated and therefore NaNs. Replace NaNs with the appropriate labels

    # Whether the present columns represents the start or end of a study
    expert_labels['Study start (Bool)'] = expert_labels['Study start/End'].notna()

    # ids same as the non-NaN id above
    expert_labels[['id']] = expert_labels[['id']].fillna(method='ffill', axis='index')

    # timestamp is the same as End if NaN
    expert_labels[['Study start/End', 'Timestamp']] = expert_labels[['Study start/End', 'Timestamp']].fillna(method='ffill', 
                                                                                                        axis='columns')
    # drop the study start end column
    expert_labels.drop(columns = ['Study start/End'], inplace = True)

    # Rename columns with large names
    expert_labels.rename(columns={
        'Background: Background: 1 = suppressed; 2 = suppression-burst  3 = continuous with periods of attenuation 4 - continuous': 'Background',        
        'Superimposed patterns:  0 - Nothing (suppressed) 1 - Seizure (convulsive or nonconvulsive); 2 - Myoclonic status; 3 - polyspike-wave; 4 -GPED; 5- non-GPED periodic patern; 6 - epileptiform discharges; 7 - nothing epileptiform. 8-GPED-Seizure': 'Superimposed patterns', 
        'Reactivity: 0- No; 1- Yes': "Reactivity"
        }, inplace=True)

    # drop rows for which background and superimposed patterns are NaNs
    expert_labels.dropna(axis = 'index', how = 'all', subset = ['Background', 'Superimposed patterns'], inplace=True)

    # Now fill NaNs in the reactivity column
    expert_labels[['Reactivity']] = expert_labels[['Reactivity']].fillna(method='ffill', axis='index')

    # Filter expert labels: focus on: background ==2 AND superimposed = one of 2, 3, 4, 5 or 8 
    # (these can be collapsed into a single category)
    # expert_labels = expert_labels.loc[(expert_labels['Background'] == 2) & (expert_labels['Superimposed patterns'].isin([2, 3, 4, 5, 8]))]

    # How do the expert labels look?
    # print('Shape of Expert labels: ', expert_labels.shape)
    # expert_labels.head(n=5)

    expert_annotations = expert_labels.apply(label_data_using_patterns, axis=1, result_type='expand')
    expert_labels.loc[:, 'Expert annotations'] = expert_annotations

    print(f'Data shape: {expert_labels.shape}')
    print(f"Data distribution: \n{Counter(expert_labels.loc[:, 'Expert annotations'])}")

    return expert_labels

def label_data_using_patterns(series):
    background = int(series['Background'])
    superimposed_patterns = int(series['Superimposed patterns'])
    
    if background == 1 and superimposed_patterns == 0: 
        return args['label_values']['suppressed']
    elif background in [3, 4]:
        return args['label_values']['normal']
    elif background == 2 and superimposed_patterns not in [0, 6, 7]:
        return args['label_values']['suppressed_with_ictal']
    elif background == 2 and superimposed_patterns in [6, 7]:
        return args['label_values']['burst_suppression']
    else:
        return None
