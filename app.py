import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Columns the uploaded CSV must have (same as the Kaggle file)
REQUIRED_COLUMNS = [
    'date', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
    'waterfront', 'view', 'condition', 'grade', 'sqft_above', 'sqft_basement',
    'yr_built', 'yr_renovated', 'zipcode', 'lat', 'long',
    'sqft_living15', 'sqft_lot15',
]

SEASON_MAP = {12: 'Winter', 1: 'Winter', 2: 'Winter',
              3: 'Spring', 4: 'Spring', 5: 'Spring',
              6: 'Summer', 7: 'Summer', 8: 'Summer',
              9: 'Autumn', 10: 'Autumn', 11: 'Autumn'}


def check_input(raw):
    """Return an error message if the file is not usable. Return None if it is fine."""
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
    if missing_cols:
        return "These columns are missing: " + ", ".join(missing_cols)
    if raw[REQUIRED_COLUMNS].isnull().any().any():
        return "The file has empty values in the required columns. Please fill or remove them."
    return None


def prepare_features(raw, prep):
    """Do the same steps as the feature engineering notebook (Milestone 2)."""
    df = raw.copy()

    # Dates
    try:
        df['date'] = pd.to_datetime(df['date'], format='%Y%m%dT%H%M%S')
    except ValueError:
        df['date'] = pd.to_datetime(df['date'])
    df['sale_year'] = df['date'].dt.year
    df['sale_month'] = df['date'].dt.month

    # New features
    df['house_age'] = (df['sale_year'] - df['yr_built']).clip(lower=0)
    df['renovation_status'] = np.where(df['yr_renovated'] > 0, 'Renovated', 'Not Renovated')
    df['season'] = df['sale_month'].map(SEASON_MAP)
    df['total_rooms'] = df['bedrooms'] + df['bathrooms']

    # Same label encoding as the notebook
    classes = prep['renovation_classes']
    df['renovation_status'] = df['renovation_status'].map({c: i for i, c in enumerate(classes)})

    # One-hot encoding
    df['zipcode'] = df['zipcode'].astype(int).astype(str)
    df = pd.get_dummies(df, columns=['season', 'zipcode'], dtype=int)

    # Scaling with the saved scaler
    df[prep['cols_to_scale']] = prep['scaler'].transform(df[prep['cols_to_scale']])

    # Same columns in the same order as training (missing ones become 0)
    return df.reindex(columns=prep['feature_columns'], fill_value=0)


@st.cache_resource
def load_files():
    model = joblib.load('model.joblib')
    prep = joblib.load('preprocessing.joblib')
    return model, prep


# ---------- App layout ----------
st.set_page_config(page_title="House Price Predictor")
st.title("House Price Predictor")
st.write("Upload a CSV file with house details. The app shows the predicted price for each house.")

try:
    model, prep = load_files()
except FileNotFoundError:
    st.error("Model files not found. Run notebooks 02 and 04 first. "
             "They create `preprocessing.joblib` and `model.joblib`.")
    st.stop()

uploaded = st.file_uploader("Upload a CSV file", type="csv")

if uploaded is None:
    st.info("Waiting for a file. You can use `data/kc_house_data.csv`.")
    st.write("The file needs these columns:")
    st.code(", ".join(REQUIRED_COLUMNS))
else:
    raw = pd.read_csv(uploaded)
    error = check_input(raw)

    if error:
        st.error(error)
    else:
        features = prepare_features(raw, prep)
        predictions = model.predict(features)

        result = raw.copy()
        result['predicted_price'] = predictions.round(0)
        if 'price' in result.columns:
            result['error'] = (result['predicted_price'] - result['price']).round(0)

        st.subheader("Predictions")
        col1, col2 = st.columns(2)
        col1.metric("Houses", f"{len(result):,}")
        col2.metric("Average predicted price", f"${result['predicted_price'].mean():,.0f}")

        first_cols = [c for c in ['id', 'price', 'predicted_price', 'error'] if c in result.columns]
        other_cols = [c for c in result.columns if c not in first_cols]
        st.dataframe(result[first_cols + other_cols])
