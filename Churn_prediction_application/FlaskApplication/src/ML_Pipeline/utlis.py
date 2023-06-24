import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score



# Function to read the data file 
def read_data(file_path, **kwargs):
    raw_data = pd.read_csv(file_path  ,**kwargs)
    return raw_data
