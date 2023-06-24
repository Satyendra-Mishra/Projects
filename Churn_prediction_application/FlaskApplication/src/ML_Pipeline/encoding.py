import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """ 
    Encodes categorical columns using LabelEncoding, OneHotEncoding and TargetEncoding.
    LabelEncoding is used for binary categorical columns
    OneHotEncoding is used for columns with <= 10 distinct values
    TargetEncoding is used for columns with higher cardinality (>10 distinct values)
    """
    def __init__(self, cols: list[str] = None, label_cols: list[str] = None, 
                 ohe_cols: list[str] = None, target_cols:list[str] = None, reduce_df: bool = False) -> None:
        """
        Parameters
        ----------
        cols : list of str -> Columns to encode.  Default is to one-hot/target/label encode all categorical columns in the DataFrame.
        reduce_df : bool
            Whether to use reduced degrees of freedom for encoding
            (that is, add N-1 one-hot columns for a column with N 
            categories). E.g. for a column with categories A, B, 
            and C: When reduce_df is True, A=[1, 0], B=[0, 1],
            and C=[0, 0].  When reduce_df is False, A=[1, 0, 0], 
            B=[0, 1, 0], and C=[0, 0, 1]
            Default = False
        """
        
        self.cols = [cols] if isinstance(cols, str) else cols
        self.label_cols = [label_cols] if isinstance(label_cols, str) else label_cols
        self.ohe_cols = [ohe_cols] if isinstance(ohe_cols, str) else ohe_cols
        self.target_cols = [target_cols] if isinstance(target_cols, str) else target_cols
        self.reduce_df = reduce_df

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Fit label/one-hot/target encoder to X and y
        Parameters
        ----------
        X : pandas DataFrame, shape [n_samples, n_columns] -> DataFrame containing columns to encode
        y : pandas Series, shape = [n_samples] -> Target values.
            
        Returns
        -------
        self : encoder
            Returns self.
        """
        
        # If columns list is not provided then get all the categorical columns from the dataframe
        if self.cols is None:
            self.cols = [col for col in X.columns if ((str(X[col].dtype) == 'object') or 
                                                      (str(X[col].dtype) == 'category'))]

        # If columns list is provided then check if columns are in X
        for col in self.cols:
            if col not in X.columns:
                raise ValueError(f"{col} not in the dataframe")
        
        # Get the columns for label Encoding
        if self.label_cols is None:
            self.label_cols = [col for col in self.cols if (X[col].nunique() <= 2)]

        # Get the columns for OneHot Encoding
        if self.ohe_cols is None:
            self.ohe_cols = [col for col in self.cols if ((X[col].nunique() > 2) and (X[col].nunique() <= 10))]

        # Get the columns for Target Encoding
        if self.target_cols is None:
            self.target_cols = [col for col in self.cols if (X[col].nunique() > 10)]

        
        # Create lable encoding mapping
        self.label_maps = dict()
        for col in self.label_cols:
            self.label_maps[col] = dict(zip(X[col].unique(), range(X[col].nunique())))

        
        # Create one hot encoding mapping
        self.ohe_maps = dict()
        for col in self.ohe_cols:
            self.ohe_maps[col] = X[col].unique().tolist()
            if self.reduce_df:
                self.ohe_maps[col].pop(-1)

        # Create target encoding mapping
        self.gloabl_target_mean = y.mean().round(4)
        self.sum_counts = dict()
        for col in self.target_cols:
            self.sum_counts[col] = dict()
            unique_categories = X[col].unique() # Numpy array
            for category in unique_categories:
                ix = X[col] == category  
                self.sum_counts[col][category] = (y[ix].sum(), ix.sum())  # numpy.int64
        
        # return the fit object
        return self

    
    
    def transform(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        """
        Perform label/one-hot/target encoding transformation.
        Parameters
        ----------
        X : pandas DataFrame, shape [n_samples, n_columns] -> DataFrame containing columns to label encode
            
        Returns
        -------
        pandas DataFrame
            Input DataFrame with transformed columns
        """
        Xo = X.copy()

        # Label Encoding transformation
        for col, label_map in self.label_maps.items():
            # map the column
            Xo[col] = Xo[col].map(label_map)
            # Fill -1 in place a any new value
            Xo[col].fillna(-1, inplace = True)

        # On Hot Encoding transformation
        for col, values in self.ohe_maps.items():
            for value in values:
                new_col = col + '_' + str(value)
                Xo[new_col] = (Xo[col] == value).astype('uint8')
            Xo.drop(col, axis = 1, inplace = True)

        # Target Encoding transformation
        if y is None:  # Meant for test set
            for col in self.sum_counts.keys():
                values = np.full(Xo.shape[0], np.nan)
                for category, sum_count in self.sum_counts[col].items():
                    ix = Xo[col] == category
                    values[ix] = (sum_count[0]/sum_count[1]).round(4)
                Xo[col] = values
                # Filling new values by global targte mean
                Xo[col].fillna(self.gloabl_target_mean, inplace = True)
        else:
            for col in self.sum_counts.keys():
                values = np.full(Xo.shape[0], np.nan)
                for category, sum_count in self.sum_counts[col].items():
                    ix = Xo[col] == category
                    if sum_count[1] > 1:
                        values[ix] = ((sum_count[0] - y[ix]) / (sum_count[1] - 1)).round(4)
                    else:
                        values[ix] = ((y.sum() - y[ix]) / (Xo.shape[0] - 1)).round(4)

                Xo[col] = values
                # Filling new values by global targte mean
                Xo[col].fillna(self.gloabl_target_mean, inplace = True)

        # Return the encoded dataframe
        return Xo
    

    def fit_transform(self, X, y=None):
        """
        Fit and transform the data via label/one-hot/target encoding.
        Parameters
        ----------
        X : pandas DataFrame, shape [n_samples, n_columns] -> DataFrame containing columns to encode
        y : pandas Series, shape = [n_samples] -> Target values (required!).

        Returns
        -------
        pandas DataFrame
            Input DataFrame with transformed columns
        """
        return self.fit(X,y).transform(X,y)






