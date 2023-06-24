import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class CustomScaler(BaseEstimator, TransformerMixin):
    """
    A custom standard scaler class with the ability to apply scaling on selected columns
    """
    def __init__(self, scale_cols: list[str] = None ) -> None:
        """
        Parameters
        ----------
        scale_cols : list of str
            Columns on which to perform scaling and normalization. Default is to scale all numerical columns
        
        """
        self.scale_cols = scale_cols

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        """
        Parameters
        ----------
        X : pandas DataFrame, shape [n_samples, n_columns]
            DataFrame containing columns to scale
        """
        # Scaling all non-categorical columns if list pf columns to scale is not provided
        if self.scale_cols is None:
            self.scale_cols = [col for col in X.columns if ((str(X[col].dtype).find('int') != 1) or 
                                                            (str(X[col].dtype).find('float') != 1))]
             
        
        # Creating the mapping
        self.maps = dict()
        for col in self.scale_cols:
            self.maps[col] = dict()
            self.maps[col]["mean"] = np.mean(X[col].to_numpy())
            self.maps[col]["std_dev"] = np.std(X[col].to_numpy())

        # Return the fit object
        return self
    
    def transform(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        """
        Parameters
        ----------
        X : pandas DataFrame, shape [n_samples, n_columns] -> DataFrame containing columns to scale

        Returns:
        --------
        Xo : pandas dataframe, shape [n_samples, n_columns] -> Dataframe with scaled columns
        """
        Xo = X.copy()

        ## Map transformation to respective columns
        for col in self.scale_cols:
            Xo[col] = (Xo[col] - self.maps[col]["mean"]) / self.maps[col]["std_dev"]
        
        # return the tranformed daraframe
        return Xo
    
    def fit_transform(self, X, y=None):
        """
        Parameters
        ----------
        X : pandas DataFrame, shape [n_samples, n_columns]
            DataFrame containing columns to scale
        """
        # Fit and return transformed dataframe
        return self.fit(X).transform(X)
    
    