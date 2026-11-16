# ============================================================
# DEMAND SPIKE DETECTION AND EARLY WARNING SYSTEM
# USING MACHINE LEARNING
# ============================================================

# ============================================================
# PART 1: IMPORT LIBRARIES
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import os

# Use a non-GUI Matplotlib backend so the script can run safely
# without Tkinter/thread errors while generating PNG files.
import matplotlib
matplotlib.use("Agg")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    silhouette_score
)

from xgboost import XGBRegressor


# ============================================================
# PART 2: PROJECT SETUP
# ============================================================

print("\n" + "=" * 70)
print("       DEMAND SPIKE DETECTION AND EARLY WARNING SYSTEM")
print("                    MACHINE LEARNING PROJECT")
print("=" * 70)

# Create output folder if it does not exist
os.makedirs("outputs", exist_ok=True)

print("\nOutput folder ready.")


# ============================================================
# PART 3: LOAD DATASET
# ============================================================

print("\n" + "=" * 70)
print("PART 3: LOADING DATASET")
print("=" * 70)

try:
    df = pd.read_csv("train.csv")
except FileNotFoundError:
    print("\nERROR: train.csv was not found.")
    print("Make sure train.csv is in the same folder as main.py.")
    exit()

print("\nDataset loaded successfully!")

print("Number of rows    :", len(df))
print("Number of columns :", len(df.columns))

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())


# ============================================================
# PART 4: DATA PREPROCESSING
# ============================================================

print("\n" + "=" * 70)
print("PART 4: DATA PREPROCESSING")
print("=" * 70)

# Convert date column
df["date"] = pd.to_datetime(df["date"])

# Sort data by store, item and date
df = df.sort_values(
    ["store", "item", "date"]
).reset_index(drop=True)

# Check missing values
print("\nMissing values:")
print(df.isnull().sum())

# Check duplicates
duplicates = df.duplicated().sum()

print("\nDuplicate records:", duplicates)

# Remove duplicates
df = df.drop_duplicates()

print("Rows after removing duplicates:", len(df))

# Check negative sales
negative_sales = (df["sales"] < 0).sum()

print("Negative sales records:", negative_sales)

# Remove negative sales if any
df = df[df["sales"] >= 0]

print("\nData preprocessing completed.")


# ============================================================
# PART 5: EXPLORATORY DATA ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("PART 5: EXPLORATORY DATA ANALYSIS")
print("=" * 70)

print("\nBasic statistical information:")
print(df[["store", "item", "sales"]].describe())


# ------------------------------------------------------------
# 5.1 TOTAL SALES
# ------------------------------------------------------------

total_sales = df["sales"].sum()

print("\nTotal sales:", total_sales)


# ------------------------------------------------------------
# 5.2 AVERAGE SALES
# ------------------------------------------------------------

average_sales = df["sales"].mean()

print("Average sales:", round(average_sales, 2))


# ------------------------------------------------------------
# 5.3 DAILY SALES
# ------------------------------------------------------------

print("\nCreating daily sales trend...")

daily_sales = (
    df.groupby("date")["sales"]
    .sum()
)

plt.figure(figsize=(14, 5))

plt.plot(
    daily_sales.index,
    daily_sales.values
)

plt.title("Daily Sales Trend")
plt.xlabel("Date")
plt.ylabel("Total Sales")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    "outputs/daily_sales_trend.png",
    dpi=300
)

plt.close()

print("Saved: outputs/daily_sales_trend.png")


# ------------------------------------------------------------
# 5.4 SALES DISTRIBUTION
# ------------------------------------------------------------

print("Creating sales distribution...")

plt.figure(figsize=(10, 5))

sns.histplot(
    df["sales"],
    bins=50,
    kde=True
)

plt.title("Sales Distribution")
plt.xlabel("Sales")
plt.ylabel("Frequency")

plt.tight_layout()

plt.savefig(
    "outputs/sales_distribution.png",
    dpi=300
)

plt.close()

print("Saved: outputs/sales_distribution.png")


# ------------------------------------------------------------
# 5.5 MONTHLY SALES
# ------------------------------------------------------------

print("Creating monthly sales graph...")

monthly_sales = (
    df.groupby(
        df["date"].dt.month
    )["sales"]
    .sum()
)

plt.figure(figsize=(10, 5))

plt.plot(
    monthly_sales.index,
    monthly_sales.values,
    marker="o"
)

plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Total Sales")

plt.xticks(range(1, 13))

plt.tight_layout()

plt.savefig(
    "outputs/monthly_sales_trend.png",
    dpi=300
)

plt.close()

print("Saved: outputs/monthly_sales_trend.png")


# ------------------------------------------------------------
# 5.6 TOP STORES
# ------------------------------------------------------------

store_sales = (
    df.groupby("store")["sales"]
    .sum()
    .sort_values(ascending=False)
)

print("\nTop 10 stores:")
print(store_sales.head(10))


plt.figure(figsize=(10, 5))

store_sales.head(10).plot(
    kind="bar"
)

plt.title("Top 10 Stores by Total Sales")
plt.xlabel("Store")
plt.ylabel("Total Sales")

plt.tight_layout()

plt.savefig(
    "outputs/top_stores.png",
    dpi=300
)

plt.close()

print("Saved: outputs/top_stores.png")


# ------------------------------------------------------------
# 5.7 TOP ITEMS
# ------------------------------------------------------------

item_sales = (
    df.groupby("item")["sales"]
    .sum()
    .sort_values(ascending=False)
)

print("\nTop 10 items:")
print(item_sales.head(10))


plt.figure(figsize=(10, 5))

item_sales.head(10).plot(
    kind="bar"
)

plt.title("Top 10 Items by Total Sales")
plt.xlabel("Item")
plt.ylabel("Total Sales")

plt.tight_layout()

plt.savefig(
    "outputs/top_items.png",
    dpi=300
)

plt.close()

print("Saved: outputs/top_items.png")


# ============================================================
# PART 6: FEATURE ENGINEERING
# ============================================================

print("\n" + "=" * 70)
print("PART 6: FEATURE ENGINEERING")
print("=" * 70)

# Date features

df["year"] = df["date"].dt.year

df["month"] = df["date"].dt.month

df["day"] = df["date"].dt.day

df["day_of_week"] = df["date"].dt.dayofweek

df["week_of_year"] = (
    df["date"]
    .dt.isocalendar()
    .week
    .astype(int)
)

# Weekend feature

df["is_weekend"] = (
    df["day_of_week"] >= 5
).astype(int)


# ------------------------------------------------------------
# LAG FEATURES
# ------------------------------------------------------------

print("\nCreating lag features...")

df["lag_1"] = (
    df.groupby(
        ["store", "item"]
    )["sales"]
    .shift(1)
)

df["lag_7"] = (
    df.groupby(
        ["store", "item"]
    )["sales"]
    .shift(7)
)

df["lag_14"] = (
    df.groupby(
        ["store", "item"]
    )["sales"]
    .shift(14)
)


# ------------------------------------------------------------
# ROLLING FEATURES
# ------------------------------------------------------------

print("Creating rolling features...")

df["rolling_mean_7"] = (
    df.groupby(
        ["store", "item"]
    )["sales"]
    .transform(
        lambda x:
        x.shift(1)
        .rolling(7)
        .mean()
    )
)

df["rolling_std_7"] = (
    df.groupby(
        ["store", "item"]
    )["sales"]
    .transform(
        lambda x:
        x.shift(1)
        .rolling(7)
        .std()
    )
)


# Remove rows containing NaN values
df = df.dropna().reset_index(drop=True)

print("\nFeature engineering completed.")

print("\nTotal records after feature engineering:")
print(len(df))


# ============================================================
# PART 7: DEFINE FEATURES AND TARGET
# ============================================================

print("\n" + "=" * 70)
print("PART 7: PREPARING MACHINE LEARNING DATA")
print("=" * 70)

features = [

    "store",

    "item",

    "year",

    "month",

    "day",

    "day_of_week",

    "week_of_year",

    "is_weekend",

    "lag_1",

    "lag_7",

    "lag_14",

    "rolling_mean_7",

    "rolling_std_7"

]

target = "sales"

X = df[features]

y = df[target]

print("\nFeatures used:")
print(features)

print("\nTarget variable:")
print(target)


# ============================================================
# PART 8: TRAIN TEST SPLIT
# ============================================================

print("\n" + "=" * 70)
print("PART 8: TRAIN TEST SPLIT")
print("=" * 70)

# Time-based split
# First 80% = training
# Last 20% = testing

split_index = int(
    len(df) * 0.80
)

X_train = X.iloc[
    :split_index
]

X_test = X.iloc[
    split_index:
]

y_train = y.iloc[
    :split_index
]

y_test = y.iloc[
    split_index:
]

print("\nTraining records:", len(X_train))

print("Testing records :", len(X_test))


# ============================================================
# PART 9: ALGORITHM 1 - LINEAR REGRESSION
# ============================================================

print("\n" + "=" * 70)
print("PART 9: ALGORITHM 1 - LINEAR REGRESSION")
print("=" * 70)

linear_model = LinearRegression()

print("\nTraining Linear Regression...")

linear_model.fit(
    X_train,
    y_train
)

linear_predictions = (
    linear_model.predict(
        X_test
    )
)

linear_mae = mean_absolute_error(
    y_test,
    linear_predictions
)

linear_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        linear_predictions
    )
)

linear_r2 = r2_score(
    y_test,
    linear_predictions
)

print("\nLinear Regression Results")

print(
    "MAE  :",
    round(linear_mae, 2)
)

print(
    "RMSE :",
    round(linear_rmse, 2)
)

print(
    "R2   :",
    round(linear_r2, 4)
)


# ============================================================
# PART 10: ALGORITHM 2 - RANDOM FOREST
# ============================================================

print("\n" + "=" * 70)
print("PART 10: ALGORITHM 2 - RANDOM FOREST")
print("=" * 70)

random_forest_model = RandomForestRegressor(

    n_estimators=100,

    max_depth=15,

    random_state=42,

    n_jobs=-1

)

print("\nTraining Random Forest...")

random_forest_model.fit(
    X_train,
    y_train
)

rf_predictions = (
    random_forest_model.predict(
        X_test
    )
)

rf_mae = mean_absolute_error(
    y_test,
    rf_predictions
)

rf_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        rf_predictions
    )
)

rf_r2 = r2_score(
    y_test,
    rf_predictions
)

print("\nRandom Forest Results")

print(
    "MAE  :",
    round(rf_mae, 2)
)

print(
    "RMSE :",
    round(rf_rmse, 2)
)

print(
    "R2   :",
    round(rf_r2, 4)
)


# ============================================================
# PART 11: ALGORITHM 3 - XGBOOST
# ============================================================

print("\n" + "=" * 70)
print("PART 11: ALGORITHM 3 - XGBOOST")
print("=" * 70)

xgb_model = XGBRegressor(

    n_estimators=300,

    learning_rate=0.05,

    max_depth=6,

    subsample=0.8,

    colsample_bytree=0.8,

    objective="reg:squarederror",

    random_state=42,

    n_jobs=-1

)

print("\nTraining XGBoost...")

xgb_model.fit(
    X_train,
    y_train
)

xgb_predictions = (
    xgb_model.predict(
        X_test
    )
)

xgb_mae = mean_absolute_error(
    y_test,
    xgb_predictions
)

xgb_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        xgb_predictions
    )
)

xgb_r2 = r2_score(
    y_test,
    xgb_predictions
)

print("\nXGBoost Results")

print(
    "MAE  :",
    round(xgb_mae, 2)
)

print(
    "RMSE :",
    round(xgb_rmse, 2)
)

print(
    "R2   :",
    round(xgb_r2, 4)
)


# ============================================================
# PART 12: MODEL COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("PART 12: MODEL COMPARISON")
print("=" * 70)

results = pd.DataFrame({

    "Algorithm": [

        "Linear Regression",

        "Random Forest",

        "XGBoost"

    ],

    "MAE": [

        linear_mae,

        rf_mae,

        xgb_mae

    ],

    "RMSE": [

        linear_rmse,

        rf_rmse,

        xgb_rmse

    ],

    "R2 Score": [

        linear_r2,

        rf_r2,

        xgb_r2

    ]

})


print("\nModel Performance:")

print(
    results.to_string(
        index=False
    )
)


# Best model

best_model_name = results.loc[
    results["RMSE"].idxmin(),
    "Algorithm"
]

print(
    "\nBest prediction model:",
    best_model_name
)


# ------------------------------------------------------------
# MODEL COMPARISON GRAPH
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))

sns.barplot(
    data=results,
    x="Algorithm",
    y="RMSE"
)

plt.title(
    "Machine Learning Model Comparison"
)

plt.xlabel("Algorithm")

plt.ylabel("RMSE")

plt.xticks(
    rotation=20
)

plt.tight_layout()

plt.savefig(
    "outputs/model_comparison.png",
    dpi=300
)

plt.close()

print(
    "\nSaved: outputs/model_comparison.png"
)


# ============================================================
# PART 13: ACTUAL VS PREDICTED SALES
# ============================================================

print("\n" + "=" * 70)
print("PART 13: ACTUAL VS PREDICTED SALES")
print("=" * 70)

# Display first 200 test records for readability

sample_size = min(
    200,
    len(y_test)
)

plt.figure(
    figsize=(14, 6)
)

plt.plot(
    y_test.values[:sample_size],
    label="Actual Sales"
)

plt.plot(
    xgb_predictions[:sample_size],
    label="XGBoost Prediction"
)

plt.title(
    "Actual vs Predicted Sales"
)

plt.xlabel(
    "Test Samples"
)

plt.ylabel(
    "Sales"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "outputs/actual_vs_predicted.png",
    dpi=300
)

plt.close()

print(
    "Saved: outputs/actual_vs_predicted.png"
)


# ============================================================
# PART 14: FEATURE IMPORTANCE
# ============================================================

print("\n" + "=" * 70)
print("PART 14: FEATURE IMPORTANCE")
print("=" * 70)

importance = pd.DataFrame({

    "Feature": features,

    "Importance":
        random_forest_model
        .feature_importances_

})

importance = (
    importance
    .sort_values(
        "Importance",
        ascending=False
    )
)

print("\nFeature importance:")

print(
    importance.to_string(
        index=False
    )
)


plt.figure(
    figsize=(10, 7)
)

sns.barplot(
    data=importance,
    x="Importance",
    y="Feature"
)

plt.title(
    "Random Forest Feature Importance"
)

plt.tight_layout()

plt.savefig(
    "outputs/feature_importance.png",
    dpi=300
)

plt.close()

print(
    "Saved: outputs/feature_importance.png"
)


# ============================================================
# PART 15: ALGORITHM 4 - ISOLATION FOREST
# ============================================================

print("\n" + "=" * 70)
print("PART 15: ALGORITHM 4 - ISOLATION FOREST")
print("=" * 70)

print(
    "\nIsolation Forest is used for anomaly detection."
)

# Use a sample for faster processing
# This is useful because the dataset is large.

anomaly_sample_size = min(
    200000,
    len(df)
)

anomaly_sample = df.sample(
    n=anomaly_sample_size,
    random_state=42
).copy()


anomaly_features = anomaly_sample[

    [

        "sales",

        "rolling_mean_7",

        "rolling_std_7",

        "lag_1",

        "lag_7"

    ]

]


print(
    "\nTraining Isolation Forest on",
    len(anomaly_sample),
    "records..."
)

isolation_model = IsolationForest(

    n_estimators=100,

    contamination=0.05,

    random_state=42,

    n_jobs=-1

)

anomaly_sample["anomaly"] = (
    isolation_model
    .fit_predict(
        anomaly_features
    )
)


# -1 = anomaly
#  1 = normal

anomaly_count = (
    anomaly_sample["anomaly"] == -1
).sum()

normal_count = (
    anomaly_sample["anomaly"] == 1
).sum()


print(
    "\nNormal records:",
    normal_count
)

print(
    "Anomalies:",
    anomaly_count
)

print(
    "Anomaly percentage:",
    round(
        anomaly_count /
        len(anomaly_sample) *
        100,
        2
    ),
    "%"
)


# ============================================================
# PART 16: DEMAND SPIKE CALCULATION
# ============================================================

print("\n" + "=" * 70)
print("PART 16: DEMAND SPIKE CALCULATION")
print("=" * 70)


anomaly_sample["spike_percentage"] = (

    (

        anomaly_sample["sales"]

        -

        anomaly_sample["rolling_mean_7"]

    )

    /

    anomaly_sample["rolling_mean_7"]

) * 100


print(
    "\nDemand spike percentage calculated."
)


# ============================================================
# PART 17: EARLY WARNING SYSTEM
# ============================================================

print("\n" + "=" * 70)
print("PART 17: EARLY WARNING SYSTEM")
print("=" * 70)


def generate_alert(row):

    increase = row[
        "spike_percentage"
    ]

    anomaly = row[
        "anomaly"
    ]

    # Critical alert
    if (
        anomaly == -1
        and increase > 30
    ):

        return "CRITICAL"

    # Warning alert
    elif increase > 10:

        return "WARNING"

    # Normal
    else:

        return "NORMAL"


anomaly_sample["alert"] = (
    anomaly_sample.apply(
        generate_alert,
        axis=1
    )
)


# ------------------------------------------------------------
# ALERT SUMMARY
# ------------------------------------------------------------

alert_counts = (
    anomaly_sample["alert"]
    .value_counts()
)

print("\nAlert Summary:")

print(alert_counts)


# ============================================================
# PART 18: DETECTED DEMAND SPIKES
# ============================================================

print("\n" + "=" * 70)
print("PART 18: DETECTED DEMAND SPIKES")
print("=" * 70)


critical_spikes = anomaly_sample[
    anomaly_sample["alert"] ==
    "CRITICAL"
].copy()


print(
    "\nNumber of critical spikes:",
    len(critical_spikes)
)


if len(critical_spikes) > 0:

    critical_spikes = (
        critical_spikes
        .sort_values(
            "spike_percentage",
            ascending=False
        )
    )

    print(
        "\nTop 20 detected demand spikes:"
    )

    print(

        critical_spikes[

            [

                "date",

                "store",

                "item",

                "sales",

                "rolling_mean_7",

                "spike_percentage",

                "alert"

            ]

        ]

        .head(20)

        .to_string(
            index=False
        )

    )

else:

    print(
        "\nNo critical demand spikes detected."
    )


# ============================================================
# PART 19: ANOMALY VISUALIZATION
# ============================================================

print("\n" + "=" * 70)
print("PART 19: ANOMALY VISUALIZATION")
print("=" * 70)

# Take last 500 records for clear graph

visual_sample = (
    anomaly_sample
    .sort_values("date")
    .tail(500)
)


normal_data = visual_sample[
    visual_sample["anomaly"] == 1
]

anomaly_data = visual_sample[
    visual_sample["anomaly"] == -1
]


plt.figure(
    figsize=(14, 6)
)


plt.scatter(

    normal_data["date"],

    normal_data["sales"],

    label="Normal"

)


plt.scatter(

    anomaly_data["date"],

    anomaly_data["sales"],

    label="Anomaly"

)


plt.title(
    "Demand Anomaly Detection using Isolation Forest"
)

plt.xlabel(
    "Date"
)

plt.ylabel(
    "Sales"
)

plt.xticks(
    rotation=45
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "outputs/anomaly_detection.png",
    dpi=300
)

plt.close()


print(
    "Saved: outputs/anomaly_detection.png"
)


# ============================================================
# PART 20: ALGORITHM 5 - K-MEANS CLUSTERING
# ============================================================

print("\n" + "=" * 70)
print("PART 20: ALGORITHM 5 - K-MEANS CLUSTERING")
print("=" * 70)


# Use sample for faster processing

cluster_sample_size = min(
    100000,
    len(df)
)

cluster_sample = df.sample(
    n=cluster_sample_size,
    random_state=42
).copy()


cluster_features = cluster_sample[

    [

        "sales",

        "rolling_mean_7",

        "rolling_std_7"

    ]

]


print(
    "\nStandardizing clustering features..."
)


scaler = StandardScaler()

scaled_features = (
    scaler.fit_transform(
        cluster_features
    )
)


print(
    "Training K-Means..."
)


kmeans_model = KMeans(

    n_clusters=3,

    random_state=42,

    n_init=10

)


cluster_sample[
    "demand_cluster"
] = (

    kmeans_model
    .fit_predict(
        scaled_features
    )

)


# ------------------------------------------------------------
# SILHOUETTE SCORE
# ------------------------------------------------------------

silhouette = silhouette_score(

    scaled_features,

    cluster_sample[
        "demand_cluster"
    ]

)


print(
    "\nSilhouette Score:",
    round(
        silhouette,
        4
    )
)


# ------------------------------------------------------------
# CLUSTER SUMMARY
# ------------------------------------------------------------

cluster_summary = (

    cluster_sample

    .groupby(
        "demand_cluster"
    )

    .agg(

        Average_Sales=(
            "sales",
            "mean"
        ),

        Average_Rolling_Mean=(
            "rolling_mean_7",
            "mean"
        ),

        Average_Std=(
            "rolling_std_7",
            "mean"
        ),

        Records=(
            "sales",
            "count"
        )

    )

)


print(
    "\nCluster Summary:"
)

print(
    cluster_summary
)


# ============================================================
# PART 21: K-MEANS VISUALIZATION
# ============================================================

print("\n" + "=" * 70)
print("PART 21: K-MEANS VISUALIZATION")
print("=" * 70)


plt.figure(
    figsize=(10, 6)
)


plt.scatter(

    cluster_sample[
        "rolling_mean_7"
    ],

    cluster_sample[
        "sales"
    ],

    c=cluster_sample[
        "demand_cluster"
    ],

    alpha=0.5

)


plt.title(
    "Demand Pattern Clustering using K-Means"
)

plt.xlabel(
    "7-Day Average Sales"
)

plt.ylabel(
    "Current Sales"
)

plt.tight_layout()


plt.savefig(
    "outputs/kmeans_clusters.png",
    dpi=300
)

plt.close()


print(
    "Saved: outputs/kmeans_clusters.png"
)


# ============================================================
# PART 22: CREATE FINAL RESULTS DATASET
# ============================================================

print("\n" + "=" * 70)
print("PART 22: CREATING FINAL RESULTS")
print("=" * 70)


# Add predictions to test data

test_results = df.iloc[
    split_index:
].copy()


test_results[
    "linear_prediction"
] = linear_predictions


test_results[
    "random_forest_prediction"
] = rf_predictions


test_results[
    "xgboost_prediction"
] = xgb_predictions


# Calculate prediction error

test_results[
    "prediction_error"
] = (

    test_results["sales"]

    -

    test_results[
        "xgboost_prediction"
    ]

)


test_results[
    "prediction_error_percentage"
] = (

    test_results[
        "prediction_error"
    ]

    /

    test_results["sales"]

) * 100


# Save

test_results.to_csv(

    "outputs/test_predictions.csv",

    index=False

)


print(
    "Saved: outputs/test_predictions.csv"
)


# ============================================================
# PART 23: SAVE MODEL COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("PART 23: SAVING MODEL RESULTS")
print("=" * 70)


results.to_csv(

    "outputs/model_comparison.csv",

    index=False

)


importance.to_csv(

    "outputs/feature_importance.csv",

    index=False

)


critical_spikes.to_csv(

    "outputs/critical_demand_spikes.csv",

    index=False

)


print(
    "Model comparison saved."
)

print(
    "Feature importance saved."
)

print(
    "Critical demand spikes saved."
)


# ============================================================
# PART 24: SAVE TRAINED MODELS
# ============================================================

print("\n" + "=" * 70)
print("PART 24: SAVING TRAINED MODELS")
print("=" * 70)


joblib.dump(

    linear_model,

    "outputs/linear_regression_model.pkl"

)


joblib.dump(

    random_forest_model,

    "outputs/random_forest_model.pkl"

)


joblib.dump(

    xgb_model,

    "outputs/xgboost_model.pkl"

)


joblib.dump(

    isolation_model,

    "outputs/isolation_forest_model.pkl"

)


joblib.dump(

    kmeans_model,

    "outputs/kmeans_model.pkl"

)


joblib.dump(

    scaler,

    "outputs/cluster_scaler.pkl"

)


print(
    "All trained models saved successfully."
)


# ============================================================
# PART 25: FINAL PROJECT SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("                 PROJECT COMPLETED")
print("=" * 70)


print("\nAlgorithms used:")

print("1. Linear Regression")
print("2. Random Forest Regressor")
print("3. XGBoost Regressor")
print("4. Isolation Forest")
print("5. K-Means Clustering")


print("\nMain tasks completed:")

print("✓ Data preprocessing")

print("✓ Exploratory data analysis")

print("✓ Feature engineering")

print("✓ Sales prediction")

print("✓ Model comparison")

print("✓ Feature importance analysis")

print("✓ Anomaly detection")

print("✓ Demand spike detection")

print("✓ Early warning system")

print("✓ Demand clustering")


print("\nBest prediction model:")

print(best_model_name)


print("\nGenerated output files:")

print("✓ daily_sales_trend.png")

print("✓ sales_distribution.png")

print("✓ monthly_sales_trend.png")

print("✓ top_stores.png")

print("✓ top_items.png")

print("✓ model_comparison.png")

print("✓ actual_vs_predicted.png")

print("✓ feature_importance.png")

print("✓ anomaly_detection.png")

print("✓ kmeans_clusters.png")

print("✓ test_predictions.csv")

print("✓ model_comparison.csv")

print("✓ feature_importance.csv")

print("✓ critical_demand_spikes.csv")


print("\n" + "=" * 70)
print("              END OF MACHINE LEARNING PROJECT")
print("=" * 70)