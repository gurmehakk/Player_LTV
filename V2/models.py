import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class ZILNModel:
    """Zero-Inflated Log-Normal model for LTV prediction"""
    
    def __init__(self, input_dim):
        self.input_dim = input_dim
        self.classification_model = None
        self.regression_model = None
        self.scaler = StandardScaler()
        
    def build_classification_model(self):
        """Build binary classification model for zero vs non-zero"""
        model = keras.Sequential([
            layers.Dense(128, activation='relu', input_shape=(self.input_dim,)),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        return model
    
    def build_regression_model(self):
        """Build regression model for positive values"""
        model = keras.Sequential([
            layers.Dense(128, activation='relu', input_shape=(self.input_dim,)),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dense(16, activation='relu'),
            layers.Dense(1, activation='linear')
        ])
        
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        return model
    
    def fit(self, X_train, y_train, X_val, y_val, epochs=50):
        """Train both classification and regression models"""
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Binary classification: zero vs non-zero
        y_binary_train = (y_train > 0).astype(int)
        y_binary_val = (y_val > 0).astype(int)
        
        print("Training classification model (zero vs non-zero)...")
        self.classification_model = self.build_classification_model()
        
        class_history = self.classification_model.fit(
            X_train_scaled, y_binary_train,
            validation_data=(X_val_scaled, y_binary_val),
            epochs=epochs,
            batch_size=64,
            verbose=0,
            callbacks=[
                keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5)
            ]
        )
        
        # Regression for positive values
        positive_mask_train = y_train > 0
        positive_mask_val = y_val > 0
        
        if positive_mask_train.sum() > 0 and positive_mask_val.sum() > 0:
            X_train_pos = X_train_scaled[positive_mask_train]
            y_train_pos = np.log1p(y_train[positive_mask_train])  # Log transform
            X_val_pos = X_val_scaled[positive_mask_val]
            y_val_pos = np.log1p(y_val[positive_mask_val])
            
            print("Training regression model (positive values)...")
            self.regression_model = self.build_regression_model()
            
            reg_history = self.regression_model.fit(
                X_train_pos, y_train_pos,
                validation_data=(X_val_pos, y_val_pos),
                epochs=epochs,
                batch_size=64,
                verbose=0,
                callbacks=[
                    keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                    keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5)
                ]
            )
            
            return class_history, reg_history
        else:
            print("No positive values for regression training")
            return class_history, None
    
    def predict(self, X):
        """Make predictions using both models"""
        X_scaled = self.scaler.transform(X)
        
        # Predict probability of non-zero
        prob_nonzero = self.classification_model.predict(X_scaled, verbose=0).flatten()
        
        # Predict values for positive cases
        if self.regression_model is not None:
            log_values = self.regression_model.predict(X_scaled, verbose=0).flatten()
            values = np.expm1(log_values)  # Inverse log transform
        else:
            values = np.ones_like(prob_nonzero)
        
        # Combine predictions
        predictions = prob_nonzero * values
        return predictions, prob_nonzero

class WhaleDetectionModel:
    """Neural Network model for whale detection"""
    
    def __init__(self, input_dim):
        self.input_dim = input_dim
        self.model = None
        self.scaler = StandardScaler()
        
    def build_model(self):
        """Build whale detection model"""
        model = keras.Sequential([
            layers.Dense(256, activation='relu', input_shape=(self.input_dim,)),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall', 'auc']
        )
        return model
    
    def fit(self, X_train, y_train, X_val, y_val, epochs=100):
        """Train whale detection model"""
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        print("Training whale detection model...")
        print(f"Class distribution - Train: {np.bincount(y_train.astype(int))}")
        print(f"Class distribution - Val: {np.bincount(y_val.astype(int))}")
        
        # Handle extreme class imbalance
        whale_count = np.sum(y_train)
        total_count = len(y_train)
        
        if whale_count < 50:  # Very few whales
            print(f"WARNING: Only {whale_count} whales in training data ({whale_count/total_count:.4%})")
            print("Using SMOTE to balance classes...")
            
            # Use SMOTE for extreme imbalance
            from imblearn.over_sampling import SMOTE
            smote = SMOTE(random_state=42, k_neighbors=min(5, whale_count-1) if whale_count > 1 else 1)
            X_train_scaled, y_train = smote.fit_resample(X_train_scaled, y_train)
            print(f"After SMOTE: {np.bincount(y_train.astype(int))}")
        
        self.model = self.build_model()
        
        # Calculate class weights
        from sklearn.utils.class_weight import compute_class_weight
        classes = np.unique(y_train)
        class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
        class_weight_dict = dict(zip(classes, class_weights))
        print(f"Class weights: {class_weight_dict}")
        
        # Use appropriate metrics for imbalanced data
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall', 'auc']
        )
        
        history = self.model.fit(
            X_train_scaled, y_train,
            validation_data=(X_val_scaled, y_val),
            epochs=epochs,
            batch_size=32,  # Smaller batch size for imbalanced data
            class_weight=class_weight_dict,
            verbose=1,  # Show progress
            callbacks=[
                keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True, monitor='val_auc', mode='max'),
                keras.callbacks.ReduceLROnPlateau(patience=10, factor=0.5, monitor='val_auc', mode='max')
            ]
        )
        
        return history
    
    def predict(self, X):
        """Make whale predictions"""
        X_scaled = self.scaler.transform(X)
        probabilities = self.model.predict(X_scaled, verbose=0).flatten()
        predictions = (probabilities > 0.5).astype(int)
        return predictions, probabilities

def plot_training_metrics(history, title="Training Metrics"):
    """Plot training and validation metrics"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(title, fontsize=16)
    
    # Loss
    axes[0, 0].plot(history.history['loss'], label='Training Loss')
    axes[0, 0].plot(history.history['val_loss'], label='Validation Loss')
    axes[0, 0].set_title('Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Accuracy (if available)
    if 'accuracy' in history.history:
        axes[0, 1].plot(history.history['accuracy'], label='Training Accuracy')
        axes[0, 1].plot(history.history['val_accuracy'], label='Validation Accuracy')
        axes[0, 1].set_title('Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
    else:
        axes[0, 1].set_title('No Accuracy Metric')
    
    # Precision
    if 'precision' in history.history:
        axes[1, 0].plot(history.history['precision'], label='Training Precision')
        axes[1, 0].plot(history.history['val_precision'], label='Validation Precision')
        axes[1, 0].set_title('Precision')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
    else:
        axes[1, 0].set_title('No Precision Metric')
    
    # Recall
    if 'recall' in history.history:
        axes[1, 1].plot(history.history['recall'], label='Training Recall')
        axes[1, 1].plot(history.history['val_recall'], label='Validation Recall')
        axes[1, 1].set_title('Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
    else:
        axes[1, 1].set_title('No Recall Metric')
    
    plt.tight_layout()
    
    # Save plot
    safe_title = title.replace(" ", "_").replace("-", "_").lower()
    plt.savefig(f'/Users/gurmehakkaur/gameramp/Player_LTV/V2/{safe_title}.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved plot: {safe_title}.png")
    plt.close()  # Close the figure to free memory
    
    # plt.show()  # Commented out to prevent blocking

def evaluate_classification_model(model, X_test, y_test, model_name="Model"):
    """Evaluate classification model with comprehensive metrics"""
    try:
        predictions, probabilities = model.predict(X_test)
        
        print(f"\n=== {model_name} Evaluation ===")
        print(f"Test set size: {len(y_test)}")
        print(f"Positive class count: {sum(y_test)}")
        print(f"Predicted positive: {sum(predictions)}")
        
        # Handle edge case with very few positives
        if sum(y_test) == 0:
            print("WARNING: No positive examples in test set")
            return predictions, probabilities
            
        print("\nClassification Report:")
        print(classification_report(y_test, predictions, zero_division=0))
        
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_test, predictions)
        print(cm)
        
        # ROC AUC - handle edge cases
        try:
            if len(set(y_test)) > 1:  # Need both classes for AUC
                auc_score = roc_auc_score(y_test, probabilities)
                print(f"\nROC AUC Score: {auc_score:.4f}")
            else:
                print("\nCannot calculate ROC AUC - only one class in test set")
        except Exception as e:
            print(f"\nCould not calculate ROC AUC: {str(e)}")
        
        # Plot confusion matrix with error handling
        try:
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'{model_name} - Confusion Matrix')
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
            
            # Save plot
            safe_name = model_name.replace(" ", "_").replace("-", "_").lower()
            plt.savefig(f'/Users/gurmehakkaur/gameramp/Player_LTV/V2/{safe_name}_confusion_matrix.png', 
                       dpi=300, bbox_inches='tight')
            print(f"✓ Saved confusion matrix: {safe_name}_confusion_matrix.png")
            plt.close()  # Close the figure to free memory
            
            # plt.show()  # Commented out to prevent blocking
        except Exception as e:
            print(f"Could not create confusion matrix plot: {str(e)}")
        
        return predictions, probabilities
        
    except Exception as e:
        print(f"Error in model evaluation: {str(e)}")
        # Return dummy values
        dummy_predictions = np.zeros(len(y_test))
        dummy_probabilities = np.zeros(len(y_test))
        return dummy_predictions, dummy_probabilities

def evaluate_regression_model(predictions, y_test, model_name="Model"):
    """Evaluate regression model with comprehensive metrics"""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    print(f"\n=== {model_name} Regression Evaluation ===")
    
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)
    
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R² Score: {r2:.4f}")
    
    # Plot predictions vs actual
    plt.figure(figsize=(10, 6))
    
    plt.subplot(1, 2, 1)
    plt.scatter(y_test, predictions, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual LTV')
    plt.ylabel('Predicted LTV')
    plt.title(f'{model_name} - Predictions vs Actual')
    
    plt.subplot(1, 2, 2)
    residuals = y_test - predictions
    plt.scatter(predictions, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted LTV')
    plt.ylabel('Residuals')
    plt.title(f'{model_name} - Residual Plot')
    
    plt.tight_layout()
    plt.close()  # Close the figure to free memory
    # plt.show()  # Commented out to prevent blocking
    
    return mae, mse, rmse, r2