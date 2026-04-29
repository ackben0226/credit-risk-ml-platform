import pandas as pd
from pathlib import Path

class MergeData:

    def __init__(self, train: pd.DataFrame, bureau_features: pd.DataFrame):
        self.train = train
        self.bureau_features = bureau_features

    def merge(self):
        merged_data = self.train.merge(
            self.bureau_features, 
            on="SK_ID_CURR",
            how="left"
        ) 
        return merged_data



