import pickle

import lightgbm as lgbm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from dateutil.relativedelta import relativedelta

def fill_misisng_values(df):
    """Fill NaN values in the 'sales' column with the mean of non-NaN values"""
    df_filled = df.copy()
    df_filled["sales"] = df_filled["sales"].fillna(df_filled["sales"].mean())
    return df_filled


def fill_missing_sales_hierarchical(df):
    """
    Fill missing sales using a hierarchical fallback strategy:

    1. Mean of:
       - same store + item + calendar date in previous years
       - same store + item + same weekday 1–2 weeks before
    2. Store + item historical mean
    3. Store-level historical mean
    4. Overall historical mean
    5. Full-dataset mean as a final fallback for very early observations

    The first four levels only use sales observations before the
    missing date. The final fallback is non-causal and is used only
    when no historical information is available.
    """

    df_filled = df.copy()

    # Ensure date is datetime
    df_filled["date"] = pd.to_datetime(df_filled["date"])

    # Sort chronologically
    df_filled = (
        df_filled
        .sort_values(["store_id", "item_id", "date"])
        .reset_index(drop=True)
    )

    # Keep original sales values.
    # This prevents imputed values from being used as historical data.
    original_sales = df_filled["sales"].copy()

    # Start with the original sales
    filled_sales = original_sales.copy()

    # Identify missing sales
    missing_indices = df_filled.index[original_sales.isna()]

    for idx in missing_indices:

        store = df_filled.loc[idx, "store_id"]
        item = df_filled.loc[idx, "item_id"]
        date = df_filled.loc[idx, "date"]

        # =========================================================
        # 1. Same store + item + historical date/weekday
        # =========================================================

        # Same calendar date in previous years
        same_date_values = original_sales[
            (df_filled["store_id"] == store)
            & (df_filled["item_id"] == item)
            & (df_filled["date"].dt.month == date.month)
            & (df_filled["date"].dt.day == date.day)
            & (df_filled["date"] < date)
        ]

        # Same weekday 1–2 weeks before
        previous_week_values = original_sales[
            (df_filled["store_id"] == store)
            & (df_filled["item_id"] == item)
            & (
                df_filled["date"].isin([
                    date - pd.Timedelta(weeks=1),
                    date - pd.Timedelta(weeks=2)
                ])
            )
        ]

        # Combine all available Level 1 observations
        level_1_values = pd.concat([
            same_date_values,
            previous_week_values
        ]).dropna()

        if len(level_1_values) > 0:
            filled_sales.loc[idx] = level_1_values.mean()
            continue

        # =========================================================
        # 2. Store + item historical mean
        # =========================================================

        store_item_values = original_sales[
            (df_filled["store_id"] == store)
            & (df_filled["item_id"] == item)
            & (df_filled["date"] < date)
        ].dropna()

        if len(store_item_values) > 0:
            filled_sales.loc[idx] = store_item_values.mean()
            continue

        # =========================================================
        # 3. Store-level historical mean
        # =========================================================

        store_values = original_sales[
            (df_filled["store_id"] == store)
            & (df_filled["date"] < date)
        ].dropna()

        if len(store_values) > 0:
            filled_sales.loc[idx] = store_values.mean()
            continue

        # =========================================================
        # 4. Overall historical mean
        # =========================================================

        historical_values = original_sales[
            df_filled["date"] < date
        ].dropna()

        if len(historical_values) > 0:
            filled_sales.loc[idx] = historical_values.mean()
            continue

        # =========================================================
        # 5. Final fallback
        # =========================================================

        # Only used when there is no historical information at all
        filled_sales.loc[idx] = original_sales.mean()

    # Apply the filled values
    df_filled["sales"] = filled_sales

    return df_filled




def correct_outliers(df, factor=3):
    """Identify and correct outliers in the 'sales' column by reducing them to the mean"""
    df_corrected = df.copy()

    # Identify outliers using z-score
    z_scores = (df_corrected["sales"] - df_corrected["sales"].mean()) / df_corrected[
        "sales"
    ].std()
    outlier_indices = np.abs(z_scores) > factor  # Adjust the threshold as needed
    # Correct outliers by reducing them to the mean
    df_corrected.loc[outlier_indices, "sales"] = df_corrected["sales"].mean()

    return df_corrected

def correct_outliers_hierarchical(df, factor=3):
    """
    Replace sales outliers using a hierarchical historical mean.

    Outliers are identified within each store-item series using a
    causal z-score based only on previous observations.

    Outlier replacement hierarchy:
    1. Mean of:
       - same store + item + calendar date in previous years
       - same store + item + same weekday 1–2 weeks before
    2. Store + item historical mean
    3. Store-level historical mean
    4. Overall historical mean
    5. Full-dataset mean as a final fallback for very early observations

    Only original sales values are used when calculating historical
    replacement values.
    """

    df_corrected = df.copy()

    # Ensure date is datetime
    df_corrected["date"] = pd.to_datetime(df_corrected["date"])

    # Sort chronologically
    df_corrected = (
        df_corrected
        .sort_values(["store_id", "item_id", "date"])
        .reset_index(drop=True)
    )

    # Keep original sales values.
    # This prevents corrected outliers from being reused
    # as historical observations.
    original_sales = df_corrected["sales"].copy()

    # =============================================================
    # Calculate causal mean and standard deviation
    # within each store-item series
    # =============================================================

    historical_mean = (
        df_corrected
        .groupby(["store_id", "item_id"])["sales"]
        .transform(
            lambda x: x.shift().expanding().mean()
        )
    )

    historical_std = (
        df_corrected
        .groupby(["store_id", "item_id"])["sales"]
        .transform(
            lambda x: x.shift().expanding().std()
        )
    )

    # Calculate causal z-score
    z_scores = (
        (df_corrected["sales"] - historical_mean)
        / historical_std
    )

    # Identify outliers
    outliers = z_scores.abs() > factor

    # =============================================================
    # Correct each outlier using hierarchical historical mean
    # =============================================================

    for idx in df_corrected.index[outliers]:

        store = df_corrected.loc[idx, "store_id"]
        item = df_corrected.loc[idx, "item_id"]
        date = df_corrected.loc[idx, "date"]

        # ---------------------------------------------------------
        # 1. Same store + item + historical date/weekday
        # ---------------------------------------------------------

        # Same calendar date from previous years
        same_date_values = original_sales[
            (df_corrected["store_id"] == store)
            & (df_corrected["item_id"] == item)
            & (df_corrected["date"].dt.month == date.month)
            & (df_corrected["date"].dt.day == date.day)
            & (df_corrected["date"] < date)
        ]

        # Same weekday 1–2 weeks before
        previous_week_values = original_sales[
            (df_corrected["store_id"] == store)
            & (df_corrected["item_id"] == item)
            & (
                df_corrected["date"].isin([
                    date - pd.Timedelta(weeks=1),
                    date - pd.Timedelta(weeks=2)
                ])
            )
        ]

        # Combine Level 1 observations
        level_1_values = pd.concat([
            same_date_values,
            previous_week_values
        ]).dropna()

        if len(level_1_values) > 0:
            df_corrected.loc[idx, "sales"] = level_1_values.mean()
            continue

        # ---------------------------------------------------------
        # 2. Store + item historical mean
        # ---------------------------------------------------------

        store_item_values = original_sales[
            (df_corrected["store_id"] == store)
            & (df_corrected["item_id"] == item)
            & (df_corrected["date"] < date)
        ].dropna()

        if len(store_item_values) > 0:
            df_corrected.loc[idx, "sales"] = store_item_values.mean()
            continue

        # ---------------------------------------------------------
        # 3. Store-level historical mean
        # ---------------------------------------------------------

        store_values = original_sales[
            (df_corrected["store_id"] == store)
            & (df_corrected["date"] < date)
        ].dropna()

        if len(store_values) > 0:
            df_corrected.loc[idx, "sales"] = store_values.mean()
            continue

        # ---------------------------------------------------------
        # 4. Overall historical mean
        # ---------------------------------------------------------

        historical_values = original_sales[
            df_corrected["date"] < date
        ].dropna()

        if len(historical_values) > 0:
            df_corrected.loc[idx, "sales"] = historical_values.mean()
            continue

        # ---------------------------------------------------------
        # 5. Final fallback
        # ---------------------------------------------------------

        # Used only when there is no historical information at all
        df_corrected.loc[idx, "sales"] = original_sales.mean()

    return df_corrected


def get_sample_stores(df: pd.DataFrame, store_id: int = 1) -> pd.DataFrame:
    """Get the sample stores with store_id"""
    grouped = df.groupby("store_id")
    sample_store = grouped.get_group((store_id))
    return sample_store


def save_data(df, file_path, file_format="feather"):
    """
    Save a DataFrame to a specified file format.

    Parameters:
    - df (pd.DataFrame): The DataFrame to be saved.
    - file_path (str): The path where the file will be saved.
    - file_format (str): The format in which to save the file. Supported formats: 'feather', 'csv'.
                        Default is 'feather'.
    Example:
    ```python
    # Assuming df is the DataFrame you want to save
    save_data(df, 'output_data.feather', file_format='feather')
    ```

    Note:
    - Make sure to have the required libraries (pandas and feather-format) installed.
    """
    if file_format.lower() == "feather":
        # Save to Feather format
        df.to_feather(file_path)
        print(f"DataFrame saved to {file_path} in Feather format.")
    elif file_format.lower() == "csv":
        # Save to CSV format
        df.to_csv(file_path, index=False)
        print(f"DataFrame saved to {file_path} in CSV format.")
    else:
        print(
            f"Error: Unsupported file format '{file_format}'. Supported formats: 'feather', 'csv'."
        )


def flatten_prophet_predictions(predictions_dict):
    all_dfs = []

    for store_item, df in predictions_dict.items():
        df = df.copy()
        df["store_item"] = store_item
        all_dfs.append(df)

    return pd.concat(all_dfs, ignore_index=True)


def load_model(file_path):
    """
    Load a machine learning model from a file.

    Parameters:
    - file_path: The file path from where the model will be loaded.

    Returns:
    - The loaded model.
    """
    try:
        with open(file_path, "rb") as file:
            model = pickle.load(file)
            print(f"Sklearn model loaded from {file_path}")

    except (pickle.UnpicklingError, FileNotFoundError):
        # If loading as scikit-learn model fails or the file is not found,
        # assume it is a LightGBM model (scikit-learn API)
        model = lgbm.Booster(model_file=file_path)
        print(f"LightGBM (scikit-learn API) model loaded from {file_path}")

    return model


# Function to calculate WAPE (Weighted Absolute Percentage Error)
def weighted_absolute_percentage_error(y_true, y_pred):
    """
    Calculate Weighted Absolute Percentage Error

    Args:
        y_true: Actual values
        y_pred: Predicted values

    Returns:
        WAPE value (percentage)
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return 100 * np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true))
