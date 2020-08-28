"""
Experiment summary
------------------
Treat each province/state in a country cases over time
as a vector, do a simple K-Nearest Neighbor between 
countries. What country has the most similar trajectory
to a given country?
"""

import sys
sys.path.insert(0, '..')

from utils import data
import os
import sklearn
import numpy as np
from sklearn.neighbors import (
    KNeighborsClassifier,
    DistanceMetric
)
from sklearn.cluster import (
    KMeans
)
import json

# ------------ HYPERPARAMETERS -------------
BASE_PATH = '../COVID-19/csse_covid_19_data/'
N_CLUSTERS = 4
MIN_CASES = 1000
NORMALIZE = True
# ------------------------------------------

confirmed = os.path.join(
    BASE_PATH, 
    'csse_covid_19_time_series',
    'time_series_covid19_confirmed_global_single.csv')
confirmed = data.load_csv_data(confirmed)
features = []
targets = []

for val in np.unique(confirmed["Country/Region"]):
    df = data.filter_by_attribute(
        confirmed, "Country/Region", val)
    cases, labels = data.get_cases_chronologically(df)
    features.append(cases)
    targets.append(labels)

features = np.concatenate(features, axis=0)
targets = np.concatenate(targets, axis=0)
predictions = {}

above_min_cases = features.sum(axis=-1) > MIN_CASES
features = features[above_min_cases]
targets = targets[above_min_cases]
if NORMALIZE:
    features = features / features.sum(axis=-1, keepdims=True)

# train kmeans
kmeans = KMeans(n_clusters=N_CLUSTERS)
kmeans.fit(features)

'''
for val in np.unique(confirmed["Country/Region"]):
    df = data.filter_by_attribute(
        confirmed, "Country/Region", val)
    cases, labels = data.get_cases_chronologically(df)
    # nearest country to this one based on trajectory
    cluster = kmeans.predict(features)
    predictions[val] = cluster[0]
'''
clusters = kmeans.predict(features)
print(clusters)
print(targets.T[1])

for i, val in enumerate(targets.T[1]):
    predictions[val] = [int(clusters[i])]

print(predictions)

with open('results/kmeans_raw.json', 'w') as f:
    json.dump(predictions, f, indent=4)
