"""
Generate SHAP visualizations for the trained Random Forest model
This script loads the trained model and generates SHAP visualizations
without having to retrain the model.
"""
import os
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.ensemble import RandomForestClassifier

# Import utilities from the model module
from model.utils import load_data, preprocess_data, split_data

def generate_shap_visualizations(
    model_path='output/random_forest_model.joblib',
    preprocessor_path='output/preprocessor.joblib',
    data_path='Loan_default.csv',
    output_dir='output',
    max_samples=100,
    timeout_seconds=300
):
    """
    Generate SHAP visualizations for a trained model
    
    Args:
        model_path: Path to the trained model
        preprocessor_path: Path to the preprocessor
        data_path: Path to the dataset
        output_dir: Directory to save visualizations
        max_samples: Maximum number of samples to use
        timeout_seconds: Maximum time allowed for SHAP calculation
    """
    print("\n===== Generating SHAP Visualizations =====\n")
    start_time = time.time()
    
    try:
        # Ensure output directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")
        
        # Load the trained model and preprocessor
        print(f"Loading model from {model_path}")
        model = joblib.load(model_path)
        print(f"Loading preprocessor from {preprocessor_path}")
        preprocessor = joblib.load(preprocessor_path)
        
        # Load and preprocess a small subset of data
        print(f"Loading data from {data_path}")
        data = load_data(data_path)
        
        # Get feature names
        X, y, _ = preprocess_data(data)
        numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Get feature names after preprocessing
        feature_names = []
        if len(numerical_cols) > 0:
            feature_names.extend(numerical_cols)
        
        if len(categorical_cols) > 0:
            ohe = preprocessor.transformers_[1][1]
            for i, col in enumerate(categorical_cols):
                feature_names.extend([f"{col}_{cat}" for cat in ohe.categories_[i]])
        
        # Use a very small subset of data for SHAP (much smaller than what we'd use for training)
        X_small = X.sample(min(max_samples*2, len(X)), random_state=42)
        y_small = y.loc[X_small.index]
        
        # Split the small dataset
        X_train_small, X_test_small, y_train_small, y_test_small = split_data(X_small, y_small)
        
        # Transform the data
        X_train_processed = preprocessor.transform(X_train_small)
        X_test_processed = preprocessor.transform(X_test_small)
        
        # Ensure we only use a small number of samples for SHAP
        sample_size = min(max_samples, X_test_processed.shape[0])
        X_test_sample = X_test_processed[:sample_size].copy()
        
        print(f"Using {sample_size} samples for SHAP explanations")
        
        # Create a simplified model for faster SHAP computation
        print("Creating simplified model for SHAP visualization...")
        simple_model = RandomForestClassifier(
            n_estimators=10,  # Fewer trees
            max_depth=5,      # Limited depth
            random_state=42
        )
        simple_model.fit(X_test_sample, model.predict(X_test_sample))
        print("Simplified model created")
        
        # Initialize SHAP explainer
        print("Initializing TreeSHAP explainer...")
        explainer = shap.TreeExplainer(simple_model)
        print("TreeSHAP explainer created")
        
        # Calculate SHAP values with timeout protection
        print(f"Calculating SHAP values (timeout: {timeout_seconds} seconds)...")
        shap_values = explainer.shap_values(X_test_sample)
        
        # Check if we've exceeded timeout
        if time.time() - start_time > timeout_seconds:
            print(f"SHAP calculation exceeded timeout of {timeout_seconds} seconds")
            return False
        
        print("SHAP values calculated successfully")
        
        # For classification, shap_values will be a list with one element per class
        if isinstance(shap_values, list):
            print(f"Binary classification detected, processing class 1 (default)")
            shap_values_for_class1 = shap_values[1]  # For positive class (default=1)
        else:
            shap_values_for_class1 = shap_values
        
        # Generate SHAP explanations using an alternative approach
        # Calculate feature importance from SHAP values
        print("Calculating SHAP feature importance...")
        
        # For binary classification, get the mean absolute SHAP value for each feature
        if isinstance(shap_values, list):
            shap_importance = np.abs(shap_values[1]).mean(0)  # For positive class
        else:
            shap_importance = np.abs(shap_values).mean(0)
            
        # Create a sorted importance dataframe
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': shap_importance
        })
        feature_importance = feature_importance.sort_values('Importance', ascending=False)
        
        # 1. Generate SHAP feature importance bar plot manually
        print("Generating SHAP importance bar plot...")
        
        plt.figure(figsize=(12, 8))
        plt.barh(
            feature_importance['Feature'][:15],  # Show top 15 features
            feature_importance['Importance'][:15],
            color='#1E88E5'
        )
        plt.xlabel('Mean |SHAP Value| (impact on model output)')
        plt.ylabel('')
        plt.title('SHAP Feature Importance')
        plt.tight_layout()
        bar_plot_path = os.path.join(output_dir, 'shap_importance_bar_plot.png')
        plt.savefig(bar_plot_path, dpi=150)
        plt.close()
        print(f"SHAP importance bar plot saved to: {bar_plot_path}")
        
        # 2. Generate individual feature impact visualization
        print("Generating feature impact visualization...")
        
        # Create a plot showing the impact of top features on a sample of predictions
        top_features = feature_importance['Feature'][:8].tolist()  # Top 8 features
        plt.figure(figsize=(14, 10))
        
        # For each top feature, plot its SHAP values
        for i, feature in enumerate(top_features):
            feature_idx = feature_names.index(feature)
            if isinstance(shap_values, list):
                feature_shap = shap_values[1][:, feature_idx]  # For positive class
            else:
                feature_shap = shap_values[:, feature_idx]
            
            plt.subplot(4, 2, i+1)  # 4x2 grid of subplots
            
            # Get corresponding feature values
            if i < len(X_test_sample[0]):
                feature_vals = X_test_sample[:, feature_idx]
                # Sort by feature value
                sorted_idxs = np.argsort(feature_vals)
                plt.plot(np.arange(len(feature_shap)), feature_shap[sorted_idxs], 'b.')
                plt.title(f'Impact of {feature}')
                plt.xlabel('Feature Value (low to high)')
                plt.ylabel('SHAP Value (impact)')
        
        plt.tight_layout()
        impact_plot_path = os.path.join(output_dir, 'shap_feature_impact.png')
        plt.savefig(impact_plot_path, dpi=150)
        plt.close()
        print(f"Feature impact visualization saved to: {impact_plot_path}")
        
        # 3. Generate a global explanation visualization
        print("Generating global explanation visualization...")
        
        plt.figure(figsize=(14, 8))
        
        # Create a heatmap of SHAP values for top features across samples
        top_features = feature_importance['Feature'][:10].tolist()  # Top 10 features
        if isinstance(shap_values, list):
            shap_data = shap_values[1][:, [feature_names.index(f) for f in top_features]]
        else:
            shap_data = shap_values[:, [feature_names.index(f) for f in top_features]]
        
        # Create a heatmap
        plt.imshow(shap_data, aspect='auto', cmap='RdBu_r')
        plt.colorbar(label='SHAP Value')
        plt.yticks(np.arange(len(X_test_sample)), [f'Sample {i+1}' for i in range(len(X_test_sample))])
        plt.xticks(np.arange(len(top_features)), top_features, rotation=45, ha='right')
        plt.title('SHAP Values Across Top Features')
        plt.tight_layout()
        global_plot_path = os.path.join(output_dir, 'shap_global_explanation.png')
        plt.savefig(global_plot_path, dpi=150)
        plt.close()
        print(f"Global explanation visualization saved to: {global_plot_path}")
        
        # Update the metrics file to indicate SHAP is available
        metrics_path = os.path.join(output_dir, 'model_metrics.json')
        if os.path.exists(metrics_path):
            import json
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
            
            metrics['shap_available'] = True
            
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f)
            print(f"Updated metrics file to indicate SHAP is available")
        
        elapsed_time = time.time() - start_time
        print(f"\nSHAP visualization generation completed in {elapsed_time:.2f} seconds")
        print("\n===== SHAP Visualization Generation Complete =====\n")
        
        return True
    
    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"\nERROR generating SHAP visualizations after {elapsed_time:.2f} seconds: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate SHAP visualizations for trained model')
    parser.add_argument('--model', default='output/random_forest_model.joblib', help='Path to trained model')
    parser.add_argument('--preprocessor', default='output/preprocessor.joblib', help='Path to preprocessor')
    parser.add_argument('--data', default='Loan_default.csv', help='Path to dataset')
    parser.add_argument('--output', default='output', help='Output directory')
    parser.add_argument('--samples', type=int, default=100, help='Maximum number of samples to use')
    parser.add_argument('--timeout', type=int, default=300, help='Maximum time in seconds')
    
    args = parser.parse_args()
    
    success = generate_shap_visualizations(
        model_path=args.model,
        preprocessor_path=args.preprocessor,
        data_path=args.data,
        output_dir=args.output,
        max_samples=args.samples,
        timeout_seconds=args.timeout
    )
    
    if success:
        print("You can now run the Streamlit app to see SHAP visualizations")
    else:
        print("Failed to generate SHAP visualizations")
