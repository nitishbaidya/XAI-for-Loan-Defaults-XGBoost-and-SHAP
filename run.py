"""
Main script to run the Loan Default Prediction XAI project
"""
import os
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description='Loan Default Prediction XAI')
    parser.add_argument('--train', action='store_true', help='Train the model and generate SHAP explanations')
    parser.add_argument('--app', action='store_true', help='Run the Streamlit app')
    
    args = parser.parse_args()
    
    if args.train:
        print("Training model and generating SHAP explanations...")
        # Add model directory to path
        sys.path.append(os.path.join(os.path.dirname(__file__), 'model'))
        from model.model_training import main as train_main
        train_main()
        print("Training completed successfully!")
    
    elif args.app:
        print("Starting Streamlit app...")
        os.system(f"streamlit run {os.path.join('app', 'streamlit_app.py')}")
    
    else:
        print("Please specify an action: --train or --app")
        print("Example: python run.py --train")
        print("Example: python run.py --app")

if __name__ == "__main__":
    main()
