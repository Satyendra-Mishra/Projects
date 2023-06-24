## Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
#matplotlib inline
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, f1_score, recall_score, confusion_matrix, classification_report
from ML_Pipeline.models import make_pipeline, evaluate_model
from ML_Pipeline.utlis import read_data
import config

#read the data
df = read_data(config.path_to_dataset)

## Keeping aside a test/holdout set
df_train_val, df_test = train_test_split(df, test_size=0.1, random_state=42)

## Splitting into train and validation set
df_train, df_val = train_test_split(df_train_val, test_size=0.12, random_state=42)

## Preparing data and a few common model parameters
y_train = df_train[config.target_var].to_numpy()
X_train = df_train.drop(columns = config.cols_to_remove + config.target_var, axis = 1)

y_val = df_val[config.target_var].to_numpy()
X_val = df_val.drop(columns = config.cols_to_remove + config.target_var, axis = 1)

weights_dict = {0 : 1.0, 1 : 3.93}
_, num_samples = np.unique(y_train, return_counts = True)
weight = (num_samples[0]/num_samples[1]).round(2)



# Equal weights to both target classes (no class imbalance correction)
lgb1 = LGBMClassifier(boosting_type = 'dart', class_weight = {0: 1, 1: 1}, min_child_samples = 20, n_jobs = - 1,
                     importance_type = 'gain', max_depth = 4, num_leaves = 31, colsample_bytree = 0.6, learning_rate = 0.1,
                     n_estimators = 21, reg_alpha = 0, reg_lambda = 0.5)

# Addressing class imbalance completely by weighting the undersampled class by the class imbalance ratio
lgb2 = LGBMClassifier(boosting_type = 'dart', class_weight = {0: 1, 1: 3.93}, min_child_samples = 20, n_jobs = - 1,
                     importance_type = 'gain', max_depth = 6, num_leaves = 63, colsample_bytree = 0.6, learning_rate = 0.1,
                     n_estimators = 201, reg_alpha = 1, reg_lambda = 1)


# Best class_weight parameter settings (partial class imbalance correction)
lgb3 = LGBMClassifier(boosting_type = 'dart', class_weight = {0: 1, 1: 3.0}, min_child_samples = 20, n_jobs = - 1,
                     importance_type = 'gain', max_depth = 6, num_leaves = 63, colsample_bytree = 0.6, learning_rate = 0.1,
                     n_estimators = 201, reg_alpha = 1, reg_lambda = 1)

## 3 different Pipeline objects for the 3 models defined above
model_1 = make_pipeline(lgb1, cols_to_scale=config.cols_to_scale, cols_to_encode=config.cols_to_encode)
model_2 = make_pipeline(lgb2, cols_to_scale=config.cols_to_scale, cols_to_encode=config.cols_to_encode)
model_3 = make_pipeline(lgb3, cols_to_scale=config.cols_to_scale, cols_to_encode=config.cols_to_encode)

## Fitting each of these models
model_1.fit(X_train, y_train.ravel())
model_2.fit(X_train, y_train.ravel())
model_3.fit(X_train, y_train.ravel())

## Getting prediction probabilities from each of these models
m1_pred_probs_trn = model_1.predict_proba(X_train)
m2_pred_probs_trn = model_2.predict_proba(X_train)
m3_pred_probs_trn = model_3.predict_proba(X_train)

## Checking correlations between the predictions of the 3 models
df_t = pd.DataFrame({'m1_pred': m1_pred_probs_trn[:,1], 'm2_pred': m2_pred_probs_trn[:,1], 'm3_pred': m3_pred_probs_trn[:,1]})
##print(df_t.shape)
##print(df_t.corr())

## Getting prediction probabilities from each of these models
m1_pred_probs_val = model_1.predict_proba(X_val)
m2_pred_probs_val = model_2.predict_proba(X_val)
m3_pred_probs_val = model_3.predict_proba(X_val)

threshold = 0.5

## Best model (Model 3) predictions
m3_preds = np.where(m3_pred_probs_val[:,1] >= threshold, 1, 0)

## Model averaging predictions (Weighted average)
m1_m2_preds = np.where(((0.1*m1_pred_probs_val[:,1]) + (0.9*m2_pred_probs_val[:,1])) >= threshold, 1, 0)

## Model 3 (Best model, tuned by GridSearch) performance on validation set
roc = roc_auc_score(y_val, m3_preds)
recall = recall_score(y_val, m3_preds)
cf = confusion_matrix(y_val, m3_preds)
##print(classification_report(y_val, m3_preds))

## Ensemble model prediction on validation set
roc = roc_auc_score(y_val, m1_m2_preds)
recall = recall_score(y_val, m1_m2_preds)
cf = confusion_matrix(y_val, m1_m2_preds)
##print(classification_report(y_val, m1_m2_preds))

### FINAL MODEL - Train final, best model ; Save model and its parameters ###
X_train = df_train.drop(columns = config.cols_to_remove + config.target_var, axis = 1)
X_val = df_val.drop(columns = config.cols_to_remove + config.target_var, axis = 1)

best_f1_lgb = LGBMClassifier(boosting_type = 'dart', class_weight = {0: 1, 1: 3.0}, min_child_samples = 20, n_jobs = - 1,
                             importance_type = 'gain', max_depth = 6, num_leaves = 63, colsample_bytree = 0.6, learning_rate = 0.1,
                             n_estimators = 201, reg_alpha = 1, reg_lambda = 1)

best_recall_lgb = LGBMClassifier(boosting_type='dart', num_leaves=31, max_depth= 6, learning_rate=0.1, n_estimators = 21,
                                 class_weight= {0: 1, 1: 3.93}, min_child_samples=2, colsample_bytree=0.6, reg_alpha=0.3,
                                 reg_lambda=1.0, n_jobs=- 1, importance_type = 'gain')


final_model = make_pipeline(best_f1_lgb, cols_to_scale=config.cols_to_scale, cols_to_encode=config.cols_to_encode)

## Fitting final model on train dataset
final_model.fit(X_train, y_train.ravel())

# Predict target probabilities
val_probs = final_model.predict_proba(X_val)[:,1]

# Predict target values on val data
val_preds = np.where(val_probs > 0.45, 1, 0) # The probability threshold can be tweaked

## Validation metrics
roc_auc_score(y_val, val_preds)
recall_score(y_val, val_preds)
confusion_matrix(y_val, val_preds)
##print(classification_report(y_val, val_preds))

## Save model object
joblib.dump(final_model, config.model_path)




## Testing


## Load model object
model = joblib.load(config.model_path)
#model = final_model
y_test = df_test[config.target_var].to_numpy()
X_test = df_test.drop(columns = config.cols_to_remove + config.target_var, axis = 1)
print(X_test.shape)
print(y_test.shape)
## Predict target probabilities
test_probs = model.predict_proba(X_test)[:,1]
## Predict target values on test data
test_preds = np.where(test_probs > 0.45, 1, 0) # Flexibility to tweak the probability threshold
#test_preds = model.predict(X_test)

## Test set metrics
roc_auc_score(y_test, test_preds)
recall_score(y_test, test_preds)
confusion_matrix(y_test, test_preds)
print(classification_report(y_test, test_preds))

## Adding predictions and their probabilities in the original test dataframe
test = df_test.copy()
test['predictions'] = test_preds
test['pred_probabilities'] = test_probs

high_churn_list = test[test.pred_probabilities > 0.7].sort_values(by = ['pred_probabilities'], ascending = False
                                                                 ).reset_index().drop(columns = ['index', 'Exited', 'predictions'], axis = 1)
print(high_churn_list.shape)
print(high_churn_list.head())
high_churn_list.to_csv('../output/high_churn_list.csv', index = False)

print("DONE")