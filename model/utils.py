import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

def load_data(file_path):
    """
    Load the dataset from the given file path
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        pandas.DataFrame: Loaded dataframe
    """
    data = pd.read_csv(file_path)
    print(f"Dataset loaded with shape: {data.shape}")
    return data

def preprocess_data(data):
    """
    Preprocess the data for model training
    
    Args:
        data (pandas.DataFrame): Raw dataframe
        
    Returns:
        tuple: X (features), y (target), preprocessor (for later use)
    """
    # Drop the LoanID column as it's just an identifier
    if 'LoanID' in data.columns:
        data = data.drop('LoanID', axis=1)
    
    # Separate features and target
    X = data.drop('Default', axis=1)
    y = data['Default']
    
    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Create preprocessor
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )
    
    # Return features, target and the preprocessor
    return X, y, preprocessor

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split the data into training and testing sets
    
    Args:
        X (pandas.DataFrame): Features
        y (pandas.Series): Target
        test_size (float): Proportion of data to use for testing
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"Training set size: {X_train.shape[0]}, Testing set size: {X_test.shape[0]}")
    return X_train, X_test, y_train, y_test

def save_model(model, preprocessor, model_path, preprocessor_path):
    """
    Save the trained model and preprocessor
    
    Args:
        model: Trained model
        preprocessor: Fitted preprocessor
        model_path (str): Path to save the model
        preprocessor_path (str): Path to save the preprocessor
    """
    joblib.dump(model, model_path)
    joblib.dump(preprocessor, preprocessor_path)
    print(f"Model saved to {model_path}")
    print(f"Preprocessor saved to {preprocessor_path}")

def load_model(model_path, preprocessor_path):
    """
    Load the trained model and preprocessor
    
    Args:
        model_path (str): Path to the saved model
        preprocessor_path (str): Path to the saved preprocessor
        
    Returns:
        tuple: Loaded model and preprocessor
    """
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    return model, preprocessor

def plot_feature_importance(importance, names, model_type, save_path=None):
    """
    Plot feature importance
    
    Args:
        importance (array): Feature importance scores
        names (list): Feature names
        model_type (str): Type of the model
        save_path (str, optional): Path to save the plot
    """
    # Create arrays from feature importance and feature names
    feature_importance = np.array(importance)
    feature_names = np.array(names)
    
    # Create a DataFrame using a Dictionary
    data = {'feature_names': feature_names, 'feature_importance': feature_importance}
    fi_df = pd.DataFrame(data)
    
    # Sort the DataFrame in order decreasing feature importance
    fi_df.sort_values(by=['feature_importance'], ascending=False, inplace=True)
    
    # Define size of bar plot
    plt.figure(figsize=(10, 8))
    
    # Plot Searborn bar chart
    sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])
    
    # Add chart labels
    plt.title(f'{model_type} Feature Importance')
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature Names')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    plt.tight_layout()
    plt.show()
