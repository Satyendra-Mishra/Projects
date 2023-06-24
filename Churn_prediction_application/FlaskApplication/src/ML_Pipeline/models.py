import numpy as np
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_score
from ML_Pipeline.encoding import CategoricalEncoder
from ML_Pipeline.feature_eng import AddFeatures
from ML_Pipeline.scaler import CustomScaler


## Automation of data preparation and model run through pipelines
def make_pipeline(model, cols_to_scale=None, cols_to_encode=None):
    '''
    Creates pipeline for the model passed as the argument. Uses standard scaling only in case of kNN models. 
    Ignores scaling step for tree/Naive Bayes models
    '''
    pipe =  Pipeline(steps = [('add_new_features', AddFeatures()),
                              ("Feature_scalinng", CustomScaler(scale_cols = cols_to_scale)),
                              ('categorical_encoding', CategoricalEncoder(cols = cols_to_encode)),
                              ('classifier', model)
                             ])
    return pipe



## Run/Evaluate all 15 models using KFold cross-validation (5 folds)
def evaluate_model(X, y, model, folds = 5, metric = 'recall'):
    pipeline = make_pipeline(model)
    scores = cross_val_score(pipeline, X, y, cv = folds, scoring = metric, n_jobs = -1)
    
    # Store results of the evaluated model
    mu, sigma = np.mean(scores), np.std(scores)
    # Printing individual model results
    print('mean = {}, std_dev = {}'.format(mu, sigma))

    return mu, sigma