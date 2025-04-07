'''
 # @ Author: Sofia Condesso
 # @ Create Time: 2025-03-03
 # @ Description: This file contains the definition of classes and methods needed to train  
 #                and evaluate the Machine Learning models that will be used in this project.
 #                
 '''

import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

class EvaluationManager:
    def __init__(self, model):
        self.model = model
        self.metrics = {}
    
    def evaluate(self, X_test, y_test):
        """Performs comprehensive model evaluation"""
        # Calculate basic metrics
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test)
        self.metrics['test_loss'] = test_loss
        self.metrics['test_accuracy'] = test_accuracy
        
        # Generate predictions
        y_pred = self.model.predict(X_test)
        
        # Calculate additional metrics
        self._calculate_advanced_metrics(y_test, y_pred)
        return self.metrics
    
    def _calculate_advanced_metrics(self, y_true, y_pred):
        """Calculates additional performance metrics"""
        from sklearn.metrics import precision_score, recall_score, f1_score
        
        self.metrics['precision'] = precision_score(y_true, y_pred.round())
        self.metrics['recall'] = recall_score(y_true, y_pred.round())
        self.metrics['f1'] = f1_score(y_true, y_pred.round())



    

class HyperparameterTuner:
    def __init__(self, X_train, y_train, X_val, y_val):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
    
    def grid_search(self, parameter_grid):
        """Performs grid search for hyperparameters"""
        best_params = None
        best_performance = float('inf')
        
        for params in self._generate_parameter_combinations(parameter_grid):
            model = self._create_model(params)
            performance = self._evaluate_parameters(model, params)
            
            if performance < best_performance:
                best_performance = performance
                best_params = params
        
        return best_params, best_performance
    
    def _generate_parameter_combinations(self, parameter_grid):
        """Generates all possible parameter combinations"""
        from itertools import product
        keys = parameter_grid.keys()
        values = parameter_grid.values()
        for instance in product(*values):
            yield dict(zip(keys, instance))


def main():
    # Load and prepare data
    data_loader = DataLoader()
    data = data_loader.load_csv("data.csv")
    X, y = data.drop('target', axis=1), data['target']
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    
    # Create and train model
    model = BaseModel(input_dim=X.shape[1])
    model.compile()
    
    trainer = TrainingManager(model.model)
    history = trainer.train(X_train, y_train, X_val, y_val)
    
    # Evaluate model
    evaluator = EvaluationManager(model.model)
    metrics = evaluator.evaluate(X_test, y_test)
    
    # Visualize results
    visualize_training(history)
    
    # Save model
    model_manager = ModelManager()
    model_manager.save_model(model.model, "my_model")

if __name__ == "__main__":
    main()