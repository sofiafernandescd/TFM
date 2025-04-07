'''
 # @ Author: Sofia Condesso
 # @ Create Time: 2025-03-03
 # @ Description: This file contains the definition of classes and methods needed to train  
 #                and evaluate the Machine Learning models that will be used in this project.
 #                
 # @ References: 
 # https://medium.com/@delija.milaim/complete-guide-to-ai-programming-templates-dd6f3ff6e943
 '''

import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os

class BaseModel:
    def __init__(self, input_dim):
        self.model = self._create_model(input_dim)
    
    def _create_model(self, input_dim):
        """Creates a basic neural network"""
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        return model
    
    def compile(self, optimizer='adam', loss='binary_crossentropy'):
        """Compiles the model"""
        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=['accuracy']
        )

class TrainingManager:
    def __init__(self, model, epochs=10, batch_size=32):
        self.model = model
        self.epochs = epochs
        self.batch_size = batch_size
        self.history = None
    
    def train(self, X_train, y_train, X_val, y_val):
        """Performs the training"""
        self.history = self.model.fit(
            X_train, y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_data=(X_val, y_val),
            callbacks=self._get_callbacks()
        )
        return self.history
    
    def _get_callbacks(self):
        """Defines callbacks for training"""
        return [
            tf.keras.callbacks.EarlyStopping(
                patience=5,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ModelCheckpoint(
                'best_weights.h5',
                save_best_only=True
            )
        ]
    

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import opensmile

# 1. Funções de Carregamento e Pré-processamento
def load_data_splits(path,test_size=0.2, random_state=42):
    #
    # Load data and extract features
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.GeMAPSv01b,
        feature_level=opensmile.FeatureLevel.Functionals,
    )
    # Get list of files
    files = [file_path.name for file_path in os.scandir(path+"/wav") if file_path.is_file()]
    feats_df = smile.process_files([path+"/wav/"+file for file in files])
    emo_db = pd.read_csv("/Users/sofiafernandes/Documents/Repos/TFM/src/emo_db_transcripts.csv")
    
    X = feats_df.values
    y = emo_db.label.values
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

# 2. Funções de Seleção de Características
def create_feature_mask(selector, X_shape):
    mask = np.zeros(X_shape[1])
    mask[selector.selected_indices] = 1
    return mask

def plot_feature_mask(mask, title):
    plt.figure(figsize=(8,6))
    grid_size = int(np.ceil(np.sqrt(len(mask))))
    mask_padded = np.pad(mask, (0, grid_size**2 - len(mask)))
    plt.imshow(mask_padded.reshape(grid_size, grid_size), cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

# 3. Função de Treinamento e Avaliação
def train_and_evaluate(model, X_train, X_test, y_train, y_test, model_name, emotions):
    # Pipeline com normalização
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', model)
    ])
    
    pipeline.fit(X_train, y_train)
    pred = pipeline.predict(X_test)
    
    # Métricas
    print(f"\n{model_name} - Relatório de Classificação:")
    print(classification_report(y_test, pred, target_names=emotions.values()))
    
    # Matriz de Confusão
    plt.figure(figsize=(8,6))
    sns.heatmap(confusion_matrix(y_test, pred), 
                annot=True, fmt='d', cmap='Blues',
                xticklabels=emotions.values(),
                yticklabels=emotions.values())
    plt.title(f'Matriz de Confusão - {model_name}')
    plt.show()
    
    return accuracy_score(y_test, pred)

# 4. Função para Modelo de Deep Learning
def create_dnn_model(input_shape, num_classes):
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_shape,)),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])
    return model

# 5. Função de Comparação Completa
def full_comparison(X_train, X_test, y_train, y_test, selector, emotions):
    # Modelos Tradicionais
    models = {
        'SVM Linear': SVC(kernel='linear'),
        'Random Forest': RandomForestClassifier(n_estimators=100),
        'Gradient Boosting': GradientBoostingClassifier()
    }
    
    # Resultados
    results = {}
    
    # Baseline sem seleção
    for name, model in models.items():
        acc = train_and_evaluate(model, X_train, X_test, y_train, y_test, 
                                f"{name} (Baseline)", emotions)
        results[f"{name} (Baseline)"] = acc
    
    # Com seleção de características
    X_train_sel = selector.transform(X_train)
    X_test_sel = selector.transform(X_test)
    
    for name, model in models.items():
        acc = train_and_evaluate(model, X_train_sel, X_test_sel, y_train, y_test,
                                f"{name} com Seleção", emotions)
        results[f"{name} com Seleção"] = acc
    
    # Deep Learning
    num_classes = len(np.unique(y_train))
    
    # Baseline DNN
    dnn_base = create_dnn_model(X_train.shape[1], num_classes)
    history_base = dnn_base.fit(StandardScaler().fit_transform(X_train), y_train,
                               epochs=50, batch_size=32,
                               validation_split=0.2, verbose=0)
    
    # DNN com Seleção
    dnn_sel = create_dnn_model(X_train_sel.shape[1], num_classes)
    history_sel = dnn_sel.fit(StandardScaler().fit_transform(X_train_sel), y_train,
                             epochs=50, batch_size=32,
                             validation_split=0.2, verbose=0)
    
    # Plotar histórico de treino
    def plot_history(history, title):
        plt.figure(figsize=(12,4))
        plt.subplot(121)
        plt.plot(history.history['accuracy'], label='Treino')
        plt.plot(history.history['val_accuracy'], label='Validação')
        plt.title(f'Acurácia - {title}')
        plt.legend()
        
        plt.subplot(122)
        plt.plot(history.history['loss'], label='Treino')
        plt.plot(history.history['val_loss'], label='Validação')
        plt.title(f'Loss - {title}')
        plt.legend()
        plt.show()
    
    plot_history(history_base, 'DNN Baseline')
    plot_history(history_sel, 'DNN com Seleção')
    
    # Avaliar modelos finais
    test_acc_base = dnn_base.evaluate(StandardScaler().fit_transform(X_test), y_test, verbose=0)[1]
    test_acc_sel = dnn_sel.evaluate(StandardScaler().fit_transform(X_test_sel), y_test, verbose=0)[1]
    
    results['DNN (Baseline)'] = test_acc_base
    results['DNN com Seleção'] = test_acc_sel
    
    # Resultados finais
    print("\nComparação Final de Acurácia:")
    for model, acc in results.items():
        print(f"{model}: {acc:.2%}")

# Exemplo de Uso
if __name__ == "__main__":
    from select_features import FeatureSelector

    # Carregar dados
    X_train, X_test, y_train, y_test = load_data_splits("/Users/sofiafernandes/.cache/kagglehub/datasets/piyushagni5/berlin-database-of-emotional-speech-emodb/versions/1")
    
    # Configurar seleção de características
    selector = FeatureSelector(algorithm='algorithm1', L=0.95)
    selector.fit(X_train)
    #selector = FeatureSelector(algorithm='algorithm2', L=0.95, MS=0.8)
    #selector.fit(X_train)
    
    # Visualizar máscara
    mask = create_feature_mask(selector, X_train.shape)
    plot_feature_mask(mask, 'Máscara de Características Selecionadas')
    
    # Executar comparação completa
    emotions = {
        'W': 'anger',
        'L': 'boredom',
        'E': 'disgust',
        'A': 'fear',
        'F': 'happiness',
        'T': 'sadness',
        'N': 'neutral'
    }
    
    full_comparison(X_train, X_test, y_train, y_test, selector, emotions)