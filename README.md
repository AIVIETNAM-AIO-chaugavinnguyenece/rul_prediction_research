# Research on Sales Forecasting
<<<<<<< HEAD
This GitHub repo is a part of AIO AI Viet Nam's Module 4 Program, a small research project relating to Sales Forecasting using ML with Explanable AI (XAI)

- **Authors:**  Vo Quang Ha, Nguyen Viet Thanh, Ngo Lam Thy, Ngo Huy Hoang, Nguyen Tran Minh Chau
=======
This GitHub repo is a part of AIO AI Viet Nam's Conquer Program, a small research project relating to Sales Forecasting using ML with Explanable AI (XAI)

- **Authors:** Ngo Huy Hoang, Vo Quang Ha, Ngo Lam Thy, Nguyen Tran Minh Chau
>>>>>>> 3aef88633fab59f5ab588631f5c2ee1e37e0416d
- **Project Type:** Research Project
- **Tech Stack:** Python, LightGBM, XGBoost, ARIMA, SARIMA, SHAP (XAI)

## Overview
- **Sales Forecasting with Explainable AI (XAI)** is a small research project leveraging Machine Learning models to forecast store-level sales with transparency and interpretability.
- The project combines time series modeling with explainability tools, and try to answer these 3 research questions:

  - RQ1 — Model family comparison: global ML vs local statistical
    Which of LightGBM, XGBoost, ARIMA and SARIMA achieves the best accuracy, and under what series conditions does each win?
  - RQ2 — Do the external factors earn their place?
    How much forecast accuracy do weather, calendar and promotion regressors contribute, and does the contribution differ by model family?
  - RQ3 — Attribution agreement and recovery of known drivers
    Do SHAP attributions from LightGBM and XGBoost agree with each other, with permutation importance, and with SARIMAX coefficients — and do they recover the item-level drivers the generator actually used?

At its core, this project explores different sales forecasting models using LightGBM, XGBoost, ARIMA, and SARIMA (optimized with Optuna), and explained using SHAP (SHapley Additive exPlanations).

## Project Structure

```bash
<<<<<<< HEAD

├── data/
│   ├── CALCE_CX2
|   ├── NASA_PCoE   
├── requirements.txt                        # Nessesary libraries
├── figures/                        
├── models/
├── notebooks/                              # Main work for PoC phase is based on Notebooks
│   ├── 01_preprocessing.ipynb              # Proprocessing notebook
│   ├── 02_EDA.ipynb                        # EDA notebook
│   ├── 03_feature_engineering.ipynb        # Feature engineer
│   ├── 04_modelling_linears.ipynb          # Model training (Linear, NLinear & DLinear)
│   ├── 05_modelling_tree_ml_models.ipynb   # Model training (Tree-based ML models in Module 3)
│   └── 06_window_length_explainable.ipynb  # Impact of Window Length & Explainable AI
=======
├── app.py                          # Streamlit web app for user interaction
├── check_data/
│   ├── check_data.xlsx             # Excel file for checking prediction
│   └── prediction_results.csv      # Model prediction output
├── data/
│   ├── sales_data.csv
│   ├── weather_data.csv
│   ├── promotion_data.csv
|   ├── holiday_data.csv
│   ├── feature_engineered_data_55_features.feather
│   ├── sales_data_preprocessed.csv
│   ├── weather_data.csv
│   └── weather_preprocessed.csv
├── docs/
│   ├── project_description_poc_phase.md  # Project detail description
│   └── shap_analysis_summary_report.md   # Quick summary of SHAP results
├── environment.yml                 # Environment for most systems
├── environment_macm1.yml           # Environment for Mac M1 chip
├── requirements.txt                # Nessesary libraries
├── figures/                        # SHAP plots and EDA visuals
├── models/
│   ├── feature_stats.json
│   └── sales_forecast_model.pkl   # Trained model
├── notebooks/                     # Main work for PoC phase is based on Notebooks
│   ├── 01_preprocessing.ipynb      # Proprocessing notebook
│   ├── 02_EDA.ipynb                # EDA notebook
│   ├── 03_feature_engineering.ipynb   # Feature engineer
│   ├── 04_modelling.ipynb          # Model training (base line: Prophet and better: Light GBM)
│   └── 05_explain_model.ipynb      # Explainable AI
├── src/                            # Modular source code
│   ├── data_loader/
│   ├── data_generator/
│   ├── ui_builder/
│   ├── ui_predictor/
│   └── utils/
>>>>>>> 3aef88633fab59f5ab588631f5c2ee1e37e0416d
└── README.md
```
