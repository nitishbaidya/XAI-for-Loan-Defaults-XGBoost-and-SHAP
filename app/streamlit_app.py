import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import joblib
import os
import sys
import seaborn as sns

# Add the parent directory to sys.path to import from model module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.utils import load_data, load_model

# Set page config
st.set_page_config(
    page_title="Loan Default Prediction XAI Dashboard",
    page_icon="🏦",
    layout="wide"
)

# Define paths
import os

# Get the absolute project root directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Define paths using absolute paths to avoid any path-related issues
DATA_PATH = os.path.join(PROJECT_ROOT, "Loan_default.csv")  # Absolute path to dataset
MODEL_PATH = os.path.join(PROJECT_ROOT, "output", "xgboost_model.joblib")
PREPROCESSOR_PATH = os.path.join(PROJECT_ROOT, "output", "preprocessor.joblib")

print(f"Looking for dataset at: {DATA_PATH}")
print(f"Project root detected as: {PROJECT_ROOT}")

# Verify dataset exists and print diagnostic info
if not os.path.exists(DATA_PATH):
    print(f"ERROR: Dataset not found at {DATA_PATH}!")
    
    # Search for the dataset in root directory with any case
    for file in os.listdir(PROJECT_ROOT):
        if file.lower().startswith('loan') and file.lower().endswith('.csv'):
            DATA_PATH = os.path.join(PROJECT_ROOT, file)
            print(f"Found alternative dataset: {DATA_PATH}")
            break

def load_assets():
    """Load all necessary assets for the app"""
    try:
        # Load data
        data = load_data(DATA_PATH)
        
        # Load model and preprocessor
        model, preprocessor = load_model(MODEL_PATH, PREPROCESSOR_PATH)
        
        # Load model metrics if available
        metrics = None
        metrics_path = os.path.join(os.path.dirname(MODEL_PATH), 'model_metrics.json')
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                import json
                metrics = json.load(f)
        
        return data, model, preprocessor, metrics
    except Exception as e:
        st.error(f"Error loading assets: {str(e)}")
        st.info("Make sure you've run the model training script first: `python run.py --train`")
        return None, None, None, None

def display_header():
    """Display the app header and introduction"""
    st.title("🏦 Loan Default Prediction with XAI")
    st.markdown("""<hr style='height:2px;border-width:0;color:#8A2BE2;background-color:#8A2BE2'>""", unsafe_allow_html=True)
    
    st.markdown("""
    This application demonstrates the use of Explainable AI (XAI) techniques to interpret 
    an XGBoost model that predicts loan defaults. The explanations are powered by SHAP 
    (SHapley Additive exPlanations).
    """)
    st.markdown("---")

def display_data_overview(data):
    """Display an overview of the dataset"""
    st.header("📊 Dataset Overview")
    st.markdown("""<hr style='height:2px;border-width:0;color:#8A2BE2;background-color:#8A2BE2'>""", unsafe_allow_html=True)
    
    # Show data statistics
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Dataset Shape")
        st.write(f"Rows: {data.shape[0]}, Columns: {data.shape[1]}")
        
        st.subheader("Default Distribution")
        default_counts = data['Default'].value_counts()
        
        # Use the same color scheme as other SHAP visualizations
        colors = ['#6CD0D0', '#FFCBDB']  # Blue for non-default, Red for default (SHAP colors)
        
        # Create pie chart with appropriate size and smaller text
        fig, ax = plt.subplots(figsize=(4, 3))
        wedges, texts, autotexts = ax.pie(
            default_counts, 
            labels=['Non-Default', 'Default'], 
            autopct='%1.1f%%',
            colors=colors,
            textprops={'fontsize': 9}  # Smaller text size
        )
        
        # Make the percentage text smaller too
        for autotext in autotexts:
            autotext.set_fontsize(8)
            
        plt.tight_layout()
        st.pyplot(fig)
    
    with col2:
        st.subheader("Sample Data")
        st.dataframe(data.head())
    
    st.markdown("---")

def display_model_performance(metrics=None):
    """Display model performance metrics"""
    st.header("📈 Model Performance")
    st.markdown("""<hr style='height:2px;border-width:0;color:#8A2BE2;background-color:#8A2BE2'>""", unsafe_allow_html=True)
    
    # Load model performance metrics if available
    try:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Model Metrics")
            if metrics:
                # Display key metrics with our updated key names
                st.metric("Accuracy", f"{metrics['accuracy']:.4f}")
                st.metric("Precision", f"{metrics['precision']:.4f}")
                st.metric("Recall", f"{metrics['recall']:.4f}")
                st.metric("F1 Score", f"{metrics['f1_score']:.4f}")
                st.metric("AUC-ROC", f"{metrics['roc_auc']:.4f}")
                st.metric("F2 Score", f"{metrics['f2_score']:.4f}")
                
                # Display threshold information
                st.subheader("Threshold Information")
                if 'threshold' in metrics:
                    st.info(f"Probability threshold used: {metrics['threshold']:.3f}")
                
                if 'optimal_threshold' in metrics:
                    st.success(f"Optimal F1 threshold: {metrics['optimal_threshold']:.3f}")
                    #if metrics['optimal_threshold'] != metrics['threshold']:
                    #    st.warning("Consider using the optimal threshold for better F1 score balance")
            else:
                st.write("No metrics data available. Please run the training script.")
        
        with col2:
            st.subheader("Confusion Matrix")
            if metrics and 'confusion_matrix' in metrics:
                # Get confusion matrix from metrics
                cm = np.array(metrics['confusion_matrix'])
                try:
                    # Get direct TP, FP, TN, FN values if available
                    tp = metrics.get('tp')
                    fp = metrics.get('fp')
                    tn = metrics.get('tn')
                    fn = metrics.get('fn')
                    
                    # If not available, extract from confusion matrix
                    if tp is None or fp is None or tn is None or fn is None:
                        tn, fp, fn, tp = cm.flatten().tolist()
                    
                    # Display confusion matrix with seaborn heatmap
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                               xticklabels=['No Default', 'Default'],
                               yticklabels=['No Default', 'Default'],
                               ax=ax)
                    plt.xlabel('Predicted')
                    plt.ylabel('Actual')
                    plt.title('Confusion Matrix')
                    st.pyplot(fig)
                    
                    # Calculate key metrics for display
                    false_positive_rate = fp / (fp + tn)
                    false_negative_rate = fn / (fn + tp)
                    
                    # Add explanation
                    st.write("""
                    **Confusion Matrix Explained:**
                    - **Top-left:** True Negatives (correctly predicted non-defaults)
                    - **Bottom-right:** True Positives (correctly predicted defaults)
                    - **Top-right:** False Positives (incorrectly predicted as defaults)
                    - **Bottom-left:** False Negatives (defaults incorrectly predicted as non-defaults)
                    """)
                except Exception as e:
                    st.error(f"Error displaying confusion matrix: {str(e)}")
                    st.exception(e)
            else:
                st.write("No confusion matrix data available.")
    
    except Exception as e:
        st.error(f"Error displaying model performance data: {str(e)}")
        st.exception(e)
    
    st.markdown("---")

def display_model_explanations(metrics=None):
    """Display model explanations using both SHAP and XGBoost feature importance"""
    st.header('🔍 Model Explanations')
    st.markdown("""<hr style='height:2px;border-width:0;color:#8A2BE2;background-color:#8A2BE2'>""", unsafe_allow_html=True)
    
    # Create tabs for different types of explanations
    tabs = st.tabs(["Feature Importance", "SHAP Explanations", "Feature Relationships", "Example Predictions"])
    
    # Tab 1: Feature Importance
    with tabs[0]:
        st.subheader('XGBoost Feature Importance')
        st.caption('Shows the most influential features in the model')
        
        # Show the SHAP importance bar plot as this is more informative
        shap_importance_path = os.path.join(PROJECT_ROOT, "output", "shap_importance_bar_plot.png")
        if os.path.exists(shap_importance_path):
            try:
                img = plt.imread(shap_importance_path)
                st.image(img, use_column_width=True)
                
                st.info("""
                This plot shows which features have the greatest impact on the model's predictions:
                - Longer bars indicate more influential features
                - Features at the top have the highest impact on predictions
                - The magnitude represents the average impact on model output
                """)
            except Exception as e:
                st.error(f"Error loading feature importance plot: {str(e)}")
        else:
            # Fallback to the regular XGBoost importance if SHAP isn't available
            xgb_importance_path = os.path.join(PROJECT_ROOT, "output", "feature_importance.png")
            if os.path.exists(xgb_importance_path):
                img = plt.imread(xgb_importance_path)
                st.image(img, use_column_width=True)
            else:    
                st.error('Feature importance plot not found.')
                
    # Tab 2: SHAP Explanations
    with tabs[1]:
        st.subheader('SHAP Feature Impact')
        st.caption('How each feature affects model predictions (positive or negative)')
        
        # Show the SHAP feature impact beeswarm plot
        feature_impact_path = os.path.join(PROJECT_ROOT, "output", "shap_feature_impact.png")
        if os.path.exists(feature_impact_path):
            img = plt.imread(feature_impact_path)
            st.image(img, use_column_width=True)
            
            st.info("""
            **How to interpret this visualization:**
            - Features are ordered by importance (most important at top)
            - Red points = positive impact (increases default risk)
            - Blue points = negative impact (decreases default risk)
            - The horizontal spread shows the magnitude of impact
            - Points represent different examples in the dataset
            """)
        else:
            st.error("SHAP feature impact plot not found.")
            
        # Also show the global impact plot which has complementary information
        st.subheader('Global Feature Impact')
        st.caption('The average impact of each feature on predictions')
            
        global_impact_path = os.path.join(PROJECT_ROOT, "output", "shap_global_explanation.png")
        if os.path.exists(global_impact_path):
            img = plt.imread(global_impact_path)
            st.image(img, use_column_width=True)
            
            st.info("""
            **Key insights:**
            - Red bars show features that increase default probability
            - Blue bars show features that decrease default probability
            - Values show the average SHAP value (impact) for each feature
            - Features with larger absolute values have more influence
            """)
        else:
            st.error("SHAP global explanation not found.")
    
    # Tab 3: Feature Relationships
    with tabs[2]:
        st.subheader('Feature Dependencies')
        st.caption('How feature values relate to their impact on predictions')
        
        dependence_path = os.path.join(PROJECT_ROOT, "output", "shap_dependence_plots.png")
        if os.path.exists(dependence_path):
            img = plt.imread(dependence_path)
            st.image(img, use_column_width=True)
            
            st.info("""
            **Understanding these plots:**
            - Each plot shows how a feature's values (x-axis) affect its impact (y-axis)
            - Red points indicate positive impact (increases default prediction)
            - Blue points indicate negative impact (decreases default prediction)
            - The trend line shows the general relationship
            - Points above the dotted line increase default risk, points below decrease it
            """)
        else:
            st.error("Feature dependency plots not found.")
                
    # Tab 4: Waterfall Plots
    with tabs[3]:
        st.subheader('Prediction Explanation')
        st.caption('How features combine to create final predictions for example cases')
        
        waterfall_path = os.path.join(PROJECT_ROOT, "output", "shap_waterfall_plots.png")
        if os.path.exists(waterfall_path):
            img = plt.imread(waterfall_path)
            st.image(img, use_column_width=True)
            
            st.info("""
            **How to read these waterfall plots:**
            - Starting from a baseline of 0.5 (neutral prediction)
            - Red connections = feature increases default probability
            - Blue connections = feature decreases default probability
            - The rightmost point shows the final prediction probability
            - Comparing different examples shows how different loan types are evaluated
            """)
        else:
            st.error("Prediction explanation plots not found.")
    
    st.markdown("---")

def make_prediction(data, model, preprocessor):
    """Allow users to make predictions on sample or custom data"""
    st.header("🧪 Make Predictions")
    st.markdown("""<hr style='height:2px;border-width:0;color:#8A2BE2;background-color:#8A2BE2'>""", unsafe_allow_html=True)
    
    st.write("Select a sample from the dataset or create a custom input to get predictions and explanations.")
    
    # Option to select sample or custom input
    input_option = st.radio("Input Method", ["Select a sample", "Custom input"])
    
    prediction_data = None
    
    if input_option == "Select a sample":
        # Let user select a sample from the dataset
        sample_index = st.slider("Select a sample index", 0, len(data)-1, 0)
        prediction_data = data.iloc[sample_index:sample_index+1].copy()
        st.dataframe(prediction_data)
    
    else:
        # Custom input form - simplified for demo
        st.subheader("Enter Custom Values")
        
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.slider("Age", 18, 80, 35)
            income = st.slider("Income ($)", 10000, 150000, 50000)
            loan_amount = st.slider("Loan Amount ($)", 5000, 150000, 30000)
            credit_score = st.slider("Credit Score", 300, 850, 650)
            months_employed = st.slider("Months Employed", 0, 120, 36)
            
        with col2:
            num_credit_lines = st.slider("Number of Credit Lines", 0, 10, 3)
            interest_rate = st.slider("Interest Rate (%)", 1.0, 25.0, 5.0)
            loan_term = st.slider("Loan Term (months)", 12, 60, 36)
            dti_ratio = st.slider("DTI Ratio", 0.1, 1.0, 0.4)
            
        # Additional categorical features
        education = st.selectbox("Education", ["High School", "Bachelor's", "Master's", "PhD"])
        employment_type = st.selectbox("Employment Type", ["Full-time", "Part-time", "Self-employed", "Unemployed"])
        marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Widowed"])
        has_mortgage = st.checkbox("Has Mortgage")
        has_dependents = st.checkbox("Has Dependents")
        loan_purpose = st.selectbox("Loan Purpose", ["Home", "Auto", "Education", "Business", "Other"])
        has_cosigner = st.checkbox("Has Co-signer")
        
        # Create a dataframe for the custom input
        custom_data = {
            'LoanID': 'CUSTOM',
            'Age': age,
            'Income': income,
            'LoanAmount': loan_amount,
            'CreditScore': credit_score,
            'MonthsEmployed': months_employed,
            'NumCreditLines': num_credit_lines,
            'InterestRate': interest_rate,
            'LoanTerm': loan_term,
            'DTIRatio': dti_ratio,
            'Education': education,
            'EmploymentType': employment_type,
            'MaritalStatus': marital_status,
            'HasMortgage': 'Yes' if has_mortgage else 'No',
            'HasDependents': 'Yes' if has_dependents else 'No',
            'LoanPurpose': loan_purpose,
            'HasCoSigner': 'Yes' if has_cosigner else 'No'
        }
        
        prediction_data = pd.DataFrame([custom_data])
        st.dataframe(prediction_data)
    
    # Make prediction if data is available
    if prediction_data is not None and st.button("Predict Default Risk"):
        try:
            # Prepare input data
            X_input = prediction_data.drop(['LoanID', 'Default'] if 'Default' in prediction_data.columns else ['LoanID'], axis=1)
            
            # Transform data
            X_processed = preprocessor.transform(X_input)
            
            # Get prediction
            prediction = model.predict(X_processed)[0]
            probability = model.predict_proba(X_processed)[0][1]
            
            # Display prediction
            st.subheader("Prediction Results")
            
            if prediction == 1:
                st.error(f"⚠️ High Risk of Default (Probability: {probability:.2f})")
            else:
                st.success(f"✅ Low Risk of Default (Probability: {probability:.2f})")
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")

def main():
    """Main function to run the Streamlit app"""
    st.sidebar.title("About")
    st.sidebar.info("""
    With the growing adoption of sophisticated machine learning (ML) and large language models (LLMs), a critical challenge has emerged—ensuring trust in AI-driven decisions. This issue is especially pronounced in high-stakes domains such as auditing and financial services, where transparency and accountability are non-negotiable. Studies have identified lack of explainability as a significant barrier to broader AI adoption in these sectors (Kokina et al., 2025; Gursoy & Cai, 2025).
    
    To address this, explainable AI (XAI) techniques are gaining traction. This project utilizes SHAP (SHapley Additive exPlanations), a model-agnostic tool grounded in cooperative game theory, to interpret predictions made by an XGBoost classifier on a loan default dataset. SHAP helps visualize and understand how individual features contribute to a model's decisions, thereby making the decision-making process more transparent and trustworthy.
    """)
    
    st.sidebar.title("References")
    st.sidebar.markdown("""
    1. Kokina, J., Blanchette, S., Davenport, T. H., & Pachamanova, D. (2025). Challenges and opportunities for artificial intelligence in auditing: Evidence from the field. *International Journal of Accounting Information Systems*, 56, 100734. https://doi.org/10.1016/j.accinf.2025.100734

    2. Gursoy, D., & Cai, R. (2025). Artificial intelligence: An overview of research trends and future directions. *International Journal of Contemporary Hospitality Management*, 37(1), 1–17. https://doi.org/10.1108/IJCHM-03-2024-0322
    """)
    
    st.sidebar.title("Technologies")
    st.sidebar.markdown("""
    - XGBoost Classifier
    - SHAP (SHapley Additive exPlanations)
    - TreeSHAP for tree-based models
    - Streamlit for the web interface
    """)
    display_header()
    

    
    # Load assets
    data, model, preprocessor, metrics = load_assets()
    
    if data is not None:
        # Display data overview
        display_data_overview(data)
        
        # Display model performance
        display_model_performance(metrics)
        
        # Display model explanations (both SHAP and XGBoost feature importance)
        display_model_explanations(metrics)
        
        # Make predictions
        if model and preprocessor:
            make_prediction(data, model, preprocessor)
    

if __name__ == "__main__":
    main()
