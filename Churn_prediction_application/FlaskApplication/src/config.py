# path to dataset
path_to_dataset = "../input/Churn_Modelling.csv"
# All features in the dataset
cols = ['RowNumber', 'CustomerId', 'Surname', 'CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 
        'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 'Exited']

# Target feature
target_var = ['Exited']

# Irrelevant features
cols_to_remove = ['RowNumber', 'CustomerId', 'Surname']
#Numerical features
num_feats = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
# Categorical features
cat_feats = ['Surname', 'Geography', 'Gender', 'HasCrCard', 'IsActiveMember']
# Features to scale
cols_to_scale = ['CreditScore', 'Age', 'Balance', 'EstimatedSalary', 'bal_per_product', 
                 'bal_by_est_salary', 'tenure_age_ratio']
# Features for numerial encoding
cols_to_encode = ['Geography', 'Gender', 'HasCrCard', 'IsActiveMember']

# saved model path
model_path = '../output/final_churn_model_f1_0_45.sav'
