'''
 # @ Author: Sofia Condesso
 # @ Create Date: 2025-03-03
 # @ Description: This file contains the definition of functions that can be reused
 #                throughout the project.
 #                
 # @ References: 
 # 
'''

import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

#################
# Data Fuctions #
#################

def split_data(X, y, test_size=0.2, validation_size=0.1):
    """Splits data into training, test, and validation sets"""
    # First split into training and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    # Then split training set into training and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=validation_size, random_state=42
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test


#####################
# Training Fuctions #
#####################

def visualize_training(history):
    """Visualizes the training progress"""
    plt.figure(figsize=(12, 4))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()


##################
# Model Fuctions #
##################

def save_model(model, path):
    """Saves the model and its weights"""
    # Save model architecture as JSON
    model_json = model.to_json()
    with open(f"{path}_architecture.json", "w") as json_file:
        json_file.write(model_json)
    
    # Save weights
    model.save_weights(f"{path}_weights.h5")
    
    # Save complete model
    model.save(f"{path}_complete.h5")
    
  
def load_model(path):
    """Loads a saved model"""
    return tf.keras.models.load_model(f"{path}_complete.h5")
    

def load_model_with_weights(path):
    """Loads model architecture and weights separately"""
    # Load architecture
    with open(f"{path}_architecture.json", "r") as json_file:
        model_json = json_file.read()
    model = tf.keras.models.model_from_json(model_json)
    
    # Load weights
    model.load_weights(f"{path}_weights.h5")
    return model


    






