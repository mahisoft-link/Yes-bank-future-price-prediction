import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="Yes Bank Stock Predictor",
    page_icon="📈",
    layout="wide"
)


# ============================================================
# LOAD MODEL
# ============================================================

import joblib

model = joblib.load(
    "notebook/yesbank_model.pkl"
)

features = joblib.load(
    "notebook/yesbank_features.pkl"
)

# ============================================================
# LOAD DATA
# ============================================================

data = pd.read_csv(
    "notebook/yesbank_processed_data.csv"
)

data["Date"] = pd.to_datetime(data["Date"])

df = data.sort_values(
    "Date"
).reset_index(drop=True)


# ============================================================
# FEATURE CREATION
# ============================================================

def create_features(data):

    data = data.copy()

    for lag in [1, 2, 3, 6, 12]:

        data[f"Close_Lag_{lag}"] = (
            data["Close"].shift(lag)
        )

    for window in [3, 6, 12]:

        data[f"Close_MA_{window}"] = (
            data["Close"]
            .rolling(window)
            .mean()
        )

        data[f"Close_STD_{window}"] = (
            data["Close"]
            .rolling(window)
            .std()
        )

    data["Return_1M"] = (
        data["Close"].pct_change()
    )

    data["Month"] = (
        data["Date"].dt.month
    )

    data["Year"] = (
        data["Date"].dt.year
    )

    return data


feature_data = create_features(data)


# ============================================================
# TITLE
# ============================================================

st.title(
    "📈 Yes Bank Stock Price Prediction"
)

st.write(
    """
    Machine Learning based forecasting system
    for predicting the next month's Yes Bank
    closing price.
    """
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "Navigation"
)

page = st.sidebar.radio(
    "Select Page",
    [
        "Dashboard",
        "Prediction",
    ]
)


# ============================================================
# DASHBOARD
# ============================================================

if page == "Dashboard":

    st.header(
        "📊 Stock Market Overview"
    )

    col1, col2, col3, col4 = st.columns(4)

    latest = df.iloc[-1]

    col1.metric(
        "Latest Close",
        f"₹{latest['Close']:.2f}"
    )

    col2.metric(
        "Latest Open",
        f"₹{latest['Open']:.2f}"
    )

    col3.metric(
        "Highest",
        f"₹{df['High'].max():.2f}"
    )

    col4.metric(
        "Lowest",
        f"₹{df['Low'].min():.2f}"
    )

    st.subheader(
        "Closing Price Trend"
    )

    chart_data = df.set_index(
        "Date"
    )[["Close"]]

    st.line_chart(
        chart_data
    )

    st.subheader(
        "Historical Data"
    )

    st.dataframe(
        df.tail(20),
        use_container_width=True
    )


# ============================================================
# PREDICTION
# ============================================================

elif page == "Prediction":

    st.header(
        "🔮 Next Month Closing Price Prediction"
    )

    latest_features = (
        feature_data
        .dropna()
        .iloc[-1]
    )

    input_data = pd.DataFrame(
        [
            [
                latest_features[col]
                for col in features
            ]
        ],
        columns=features
    )

    if st.button(
        "Predict Next Month Close"
    ):

        prediction = model.predict(
            input_data
        )[0]

        st.success(
            f"Predicted Next Month Closing Price: ₹{prediction:.2f}"
        )

        current_price = df.iloc[-1]["Close"]

        change = (
            (prediction - current_price)
            / current_price
        ) * 100

        st.metric(
            "Expected Change",
            f"{change:.2f}%"
        )

        st.warning(
            """
            This prediction is for educational/
            analytical purposes only and should
            not be treated as financial advice.
            """
        )
