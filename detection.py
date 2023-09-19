import pandas as pd
import numpy as np

from sklearn import preprocessing
from sklearn.model_selection import train_test_split

class DetectionData:
    def __init__(self) -> None:
        pass
    
    def load_data(self, str_path:str):
        print("Loading data...")
    
    
if __name__ == "__main__":
    DD = DetectionData()
    DD.load_data("data.csv")