import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score



# Function to read the data file 
def read_data(file_path, **kwargs):
    raw_data = pd.read_csv(file_path  ,**kwargs)
    return raw_data


# Cross validation evaluation
def evaluate_models(X, y, models, folds = 5, metric = 'recall'):
    results = dict()
    for name, model in models.items():
        # Evaluate model through automated pipelines
        pipeline = make_pipeline(model)
        scores = cross_val_score(pipeline, X, y, cv = folds, scoring = metric, n_jobs = -1)
        
        # Store results of the evaluated model
        results[name] = scores
        mu, sigma = np.mean(scores), np.std(scores)
        # Printing individual model results
        print('Model {}: mean = {}, std_dev = {}'.format(name, mu, sigma))
    
    return results
