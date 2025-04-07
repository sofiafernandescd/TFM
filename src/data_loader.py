'''
 # @ Author: Sofia Condesso
 # @ Create Time: 2025-03-03
 # @ Description: This file contains the definition of classes and methods needed to train  
 #                and evaluate the Machine Learning models that will be used in this project.
 '''

import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

class DataLoader:
    def __init__(self):
        self.scaler = StandardScaler()
    
    def load_csv(self, filepath):
        """Loads data from a CSV file"""
        return pd.read_csv(filepath)
    
    def load_excel(self, filepath):
        """Loads data from an Excel file"""
        return pd.read_excel(filepath)
    
    def preprocess(self, data):
        """Basic data preprocessing"""
        # Handle missing values
        data = data.fillna(data.mean())
        # Normalization
        return self.scaler.fit_transform(data)