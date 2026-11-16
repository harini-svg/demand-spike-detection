from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import os


# ============================================================
# FLASK APPLICATION
# ============================================================

app = Flask(__name__)
CORS(app)


# ============================================================
# PROJECT FOLDERS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

OUTPUT_FOLDER = os.path.join(
    BASE_DIR,
    "outputs"
)

FRONTEND_FOLDER = os.path.join(
    BASE_DIR,
    "frontend"
)


# ============================================================
# READ CSV FUNCTION
# ============================================================

def read_csv(filename):

    path = os.path.join(
        OUTPUT_FOLDER,
        filename
    )

    if not os.path.exists(path):

        return None

    df = pd.read_csv(path)

    df = df.where(
        pd.notnull(df),
        None
    )

    return df


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def home():

    return send_from_directory(
        FRONTEND_FOLDER,
        "index.html"
    )


# ============================================================
# MODEL API
# ============================================================

@app.route("/api/models")
def models():

    df = read_csv(
        "model_comparison.csv"
    )

    if df is None:

        return jsonify({
            "error":
            "model_comparison.csv not found"
        }), 404

    best = df.loc[
        df["RMSE"].idxmin()
    ]

    return jsonify({

        "models":
            df.to_dict(
                orient="records"
            ),

        "best_algorithm":
            best["Algorithm"],

        "best_rmse":
            float(best["RMSE"]),

        "best_mae":
            float(best["MAE"]),

        "best_r2":
            float(best["R2 Score"])

    })


# ============================================================
# PREDICTIONS API
# ============================================================

@app.route("/api/predictions")
def predictions():

    df = read_csv(
        "test_predictions.csv"
    )

    if df is None:

        return jsonify({
            "error":
            "test_predictions.csv not found"
        }), 404

    return jsonify(
        df.to_dict(
            orient="records"
        )
    )


# ============================================================
# FEATURE IMPORTANCE API
# ============================================================

@app.route("/api/features")
def features():

    df = read_csv(
        "feature_importance.csv"
    )

    if df is None:

        return jsonify({
            "error":
            "feature_importance.csv not found"
        }), 404

    return jsonify(
        df.to_dict(
            orient="records"
        )
    )


# ============================================================
# DEMAND SPIKES API
# ============================================================

@app.route("/api/spikes")
def spikes():

    df = read_csv(
        "critical_demand_spikes.csv"
    )

    if df is None:

        return jsonify({
            "error":
            "critical_demand_spikes.csv not found"
        }), 404

    return jsonify(
        df.to_dict(
            orient="records"
        )
    )


# ============================================================
# SUMMARY API
# ============================================================

@app.route("/api/summary")
def summary():

    result = {}

    # --------------------------------------------------------
    # TRAINING DATA SUMMARY
    # --------------------------------------------------------

    train_path = os.path.join(
        BASE_DIR,
        "train.csv"
    )

    if os.path.exists(train_path):

        df = pd.read_csv(
            train_path
        )

        if "sales" in df.columns:

            result["total_sales"] = float(
                df["sales"].sum()
            )

            result["average_sales"] = float(
                df["sales"].mean()
            )

            result["total_records"] = int(
                len(df)
            )


    # --------------------------------------------------------
    # MODEL SUMMARY
    # --------------------------------------------------------

    model_df = read_csv(
        "model_comparison.csv"
    )

    if model_df is not None:

        best = model_df.loc[
            model_df["RMSE"].idxmin()
        ]

        result["best_algorithm"] = (
            best["Algorithm"]
        )

        result["best_rmse"] = float(
            best["RMSE"]
        )

        result["best_mae"] = float(
            best["MAE"]
        )

        result["best_r2"] = float(
            best["R2 Score"]
        )


    # --------------------------------------------------------
    # CRITICAL SPIKES
    # --------------------------------------------------------

    spike_df = read_csv(
        "critical_demand_spikes.csv"
    )

    if spike_df is not None:

        result["critical_spikes"] = int(
            len(spike_df)
        )

    else:

        result["critical_spikes"] = 0


    return jsonify(result)


# ============================================================
# OUTPUT FILES
# ============================================================

@app.route(
    "/outputs/<path:filename>"
)
def output_files(filename):

    return send_from_directory(
        OUTPUT_FOLDER,
        filename
    )


# ============================================================
# RUN FLASK APPLICATION
# ============================================================

if __name__ == "__main__":

    print(
        "Open http://127.0.0.1:5000"
    )

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )