import numpy as np
import joblib
import config


def predict(df):
    try:
        if 'Exited' in list(df.columns):
            df.drop(columns = ['Exited'], axis=1, inplace=True)
        #     return "Target Variable is in the test data, prompting to check the dataset"
        for col in config.cols_to_remove:
            if col  in list(df.columns):
                df.drop(col, axis=1, inplace=True)
        ## Load model object
        model = joblib.load(config.model_path)
        ## Predict target probabilities
        test_probs = model.predict_proba(df)[:,1]
        ## Predict target values on test data
        test_preds = np.where(test_probs > 0.45, 1, 0) # Flexibility to tweak the probability threshold

        test = df.copy()
        test['predictions'] = test_preds
        test['pred_probabilities'] = test_probs

        high_churn_list = test[test.pred_probabilities > 0.7].sort_values(by = ['pred_probabilities'], ascending = False
                                                                        ).reset_index().drop(columns = ['index', 'predictions'], axis = 1)
        print(high_churn_list.shape)
        print(high_churn_list.head())
        
        return 200, high_churn_list
    except Exception as error:
        return 500, str(error)

