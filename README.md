# Loan Default Prediction with XAI

This project implements a XGBoost classifier to predict loan defaults using the Loan_default.csv dataset. It incorporates Explainable AI (XAI) techniques using SHAP (SHapley Additive exPlanations) to interpret the model's predictions.

## Features

- Machine learning pipeline for loan default prediction
- XGBoost implementation
- SHAP/TreeSHAP explainability for model interpretability
- Streamlit web application for interactive model exploration

## Project Structure

```
.
├── README.md
├── requirements.txt
├── Loan_default.csv (dataset)
├── model/
│   ├── model_training.py (model training and SHAP analysis)
│   └── utils.py (utility functions)
└── app/
    └── streamlit_app.py (Streamlit application)
```

## XAI Techniques

With the growing adoption of sophisticated machine learning (ML) and large language models (LLMs), a critical challenge has emerged—ensuring trust in AI-driven decisions. This issue is especially pronounced in high-stakes domains such as auditing and financial services, where transparency and accountability are non-negotiable. Studies have identified lack of explainability as a significant barrier to broader AI adoption in these sectors (Kokina et al., 2025; Gursoy & Cai, 2025).

To address this, explainable AI (XAI) techniques are gaining traction. This project utilizes SHAP (SHapley Additive exPlanations), a model-agnostic tool grounded in cooperative game theory, to interpret predictions made by an XGBoost classifier on a loan default dataset. SHAP helps visualize and understand how individual features contribute to a model’s decisions, thereby making the decision-making process more transparent and trustworthy.

References

 1. Kokina, J., Blanchette, S., Davenport, T. H., & Pachamanova, D. (2025). Challenges and opportunities for artificial intelligence in auditing: Evidence from the field. International Journal of Accounting Information Systems, 56, 100734. https://doi.org/10.1016/j.accinf.2025.100734

 2. Gursoy, D., & Cai, R. (2025). Artificial intelligence: An overview of research trends and future directions. International Journal of Contemporary Hospitality Management, 37(1), 1–17. https://doi.org/10.1108/IJCHM-03-2024-0322

## Dataset

The dataset contains information about loan applicants including demographic information, financial details, and whether they defaulted on their loan
It has been downloaded from Kaggle: https://www.kaggle.com/datasets/nikhil1e9/loan-default