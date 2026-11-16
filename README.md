# Demand Spike Detection and Early Warning System

A machine learning-based project designed to analyse historical sales data, identify unusual demand patterns, and generate early warnings for potential demand spikes.

## Project Overview

This project uses historical sales data to study demand trends and detect unusual changes in sales activity. It combines predictive modelling, anomaly detection, and clustering techniques with a dashboard for visualising results.

## Key Features

- Historical sales data analysis and preprocessing.
- Feature engineering using date-based, lag, and rolling statistical features.
- Sales prediction using multiple machine learning models.
- Anomaly detection to identify unusual sales patterns.
- Clustering to group similar demand patterns.
- Visualisation of sales trends, model comparisons, and detected demand spikes.
- Flask-based API to provide dashboard data.

## Machine Learning Models

- Linear Regression
- Random Forest Regressor
- XGBoost Regressor
- Isolation Forest
- K-Means Clustering

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- Flask
- HTML, CSS and JavaScript

## Project Structure

```text
demandspikeproject/
│
├── main.py
├── api.py
├── train.csv
├── test.csv
├── sample_submission.csv
│
├── frontend/
│   └── index.html
│
└── outputs/
    ├── Model files
    ├── Prediction results
    ├── Demand spike analysis
    └── Visualisation charts
```

## How It Works

1. Loads and preprocesses historical sales data.
2. Performs exploratory data analysis.
3. Creates date-based, lag, and rolling statistical features.
4. Trains and evaluates machine learning models.
5. Applies anomaly detection and clustering techniques.
6. Generates prediction results, analysis files, and visualisations.
7. Uses a Flask API to provide data to the dashboard.

## Running the Project

Install the required Python libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost flask joblib
```

Run the machine learning pipeline:

```bash
python main.py
```

Start the Flask API:

```bash
python api.py
```

Open the dashboard through the local Flask server at the address shown in the terminal.

## Note

The large trained Random Forest model file and test prediction CSV are excluded from this GitHub upload because of file-size limitations. The original files are retained separately.

## Project Status

Completed
