"""
Retail Demand Intelligence & Sales Forecasting
Final Data Analytics with AI Project

Dataset:
Kaggle Store Item Demand Forecasting Challenge
https://www.kaggle.com/competitions/demand-forecasting-kernels-only/data

Expected train.csv columns:
date, store, item, sales

Run:
    streamlit run app.py
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.set_page_config(page_title="Retail Demand Intelligence", page_icon="📈", layout="wide")

DATA_FILE = "train.csv"

@st.cache_data
def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} was not found. Download train.csv from the dataset link "
            "in README.md and place it beside app.py."
        )
    df = pd.read_csv(path, parse_dates=["date"])
    required = {"date", "store", "item", "sales"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    df = df.dropna(subset=["date", "store", "item", "sales"]).copy()
    df["sales"] = pd.to_numeric(df["sales"], errors="coerce")
    return df.dropna(subset=["sales"])

def make_daily_series(df):
    return (df.groupby("date", as_index=False)["sales"]
              .sum().sort_values("date").reset_index(drop=True))

def add_time_features(daily):
    out = daily.copy()
    out["year"] = out["date"].dt.year
    out["month"] = out["date"].dt.month
    out["day"] = out["date"].dt.day
    out["day_of_week"] = out["date"].dt.dayofweek
    out["week_of_year"] = out["date"].dt.isocalendar().week.astype(int)
    out["day_of_year"] = out["date"].dt.dayofyear
    out["lag_1"] = out["sales"].shift(1)
    out["lag_7"] = out["sales"].shift(7)
    out["lag_14"] = out["sales"].shift(14)
    out["lag_28"] = out["sales"].shift(28)
    out["rolling_7"] = out["sales"].shift(1).rolling(7).mean()
    out["rolling_28"] = out["sales"].shift(1).rolling(28).mean()
    return out

FEATURES = [
    "year", "month", "day", "day_of_week", "week_of_year",
    "day_of_year", "lag_1", "lag_7", "lag_14", "lag_28",
    "rolling_7", "rolling_28"
]

@st.cache_resource
def train_model(daily):
    model_df = add_time_features(daily).dropna().reset_index(drop=True)
    split = int(len(model_df) * 0.80)
    train_df = model_df.iloc[:split]
    valid_df = model_df.iloc[split:]

    model = RandomForestRegressor(
        n_estimators=150, max_depth=14, min_samples_leaf=2,
        random_state=42, n_jobs=-1
    )
    model.fit(train_df[FEATURES], train_df["sales"])
    pred = np.maximum(model.predict(valid_df[FEATURES]), 0)

    metrics = {
        "MAE": mean_absolute_error(valid_df["sales"], pred),
        "RMSE": mean_squared_error(valid_df["sales"], pred) ** 0.5
    }
    validation = valid_df[["date", "sales"]].copy()
    validation["predicted_sales"] = pred
    return model, metrics, validation

def forecast_next_days(daily, model, horizon):
    history = daily[["date", "sales"]].copy().sort_values("date").reset_index(drop=True)
    forecasts = []
    for _ in range(horizon):
        next_date = history["date"].iloc[-1] + pd.Timedelta(days=1)
        temp = pd.concat(
            [history, pd.DataFrame({"date": [next_date], "sales": [np.nan]})],
            ignore_index=True
        )
        feat = add_time_features(temp).iloc[[-1]][FEATURES]
        prediction = float(max(model.predict(feat)[0], 0))
        forecasts.append({"date": next_date, "forecast_sales": prediction})
        history = pd.concat(
            [history, pd.DataFrame({"date": [next_date], "sales": [prediction]})],
            ignore_index=True
        )
    return pd.DataFrame(forecasts)

st.title("📈 Retail Demand Intelligence & Sales Forecasting")
st.caption("Historical demand → trends → forecasting → business decision support")

try:
    df = load_data(DATA_FILE)
except Exception as exc:
    st.error(str(exc))
    st.info("Download the dataset and save the file as train.csv in the project folder.")
    st.stop()

daily = make_daily_series(df)

total_sales = df["sales"].sum()
avg_daily_sales = daily["sales"].mean()
peak_day = daily.loc[daily["sales"].idxmax()]
stores = df["store"].nunique()
items = df["item"].nunique()

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Units Sold", f"{total_sales:,.0f}")
c2.metric("Average Daily Demand", f"{avg_daily_sales:,.0f}")
c3.metric("Peak Daily Demand", f"{peak_day['sales']:,.0f}")
c4.metric("Stores", f"{stores:,}")
c5.metric("Items", f"{items:,}")

st.divider()

st.sidebar.header("Controls")
horizon = st.sidebar.slider("Forecast horizon (days)", 7, 30, 14)
store_filter = st.sidebar.selectbox("Store view", ["All stores"] + sorted(df["store"].unique().tolist()))

if store_filter != "All stores":
    filtered = df[df["store"] == store_filter]
    view_daily = filtered.groupby("date", as_index=False)["sales"].sum()
else:
    view_daily = daily.copy()

st.subheader("1. Sales Trend")
fig = px.line(view_daily, x="date", y="sales", title="Daily Sales Trend",
              labels={"date": "Date", "sales": "Units Sold"})
st.plotly_chart(fig, use_container_width=True)

left, right = st.columns(2)
with left:
    st.subheader("2. Store Performance")
    store_sales = df.groupby("store", as_index=False)["sales"].sum().sort_values("sales", ascending=False)
    fig_store = px.bar(store_sales.head(10), x="store", y="sales",
                       title="Top Stores by Total Sales",
                       labels={"store": "Store", "sales": "Units Sold"})
    st.plotly_chart(fig_store, use_container_width=True)

with right:
    st.subheader("3. Item Performance")
    item_sales = df.groupby("item", as_index=False)["sales"].sum().sort_values("sales", ascending=False)
    fig_item = px.bar(item_sales.head(10), x="item", y="sales",
                      title="Top 10 Items by Total Sales",
                      labels={"item": "Item", "sales": "Units Sold"})
    st.plotly_chart(fig_item, use_container_width=True)

st.subheader("4. Forecasting Model")
st.write("Random Forest regression with calendar, lag and rolling-demand features and a chronological 80/20 validation split.")

with st.spinner("Training forecasting model..."):
    model, metrics, validation = train_model(daily)

m1, m2 = st.columns(2)
m1.metric("Validation MAE", f"{metrics['MAE']:,.2f}")
m2.metric("Validation RMSE", f"{metrics['RMSE']:,.2f}")

fig_val = px.line(validation, x="date", y=["sales", "predicted_sales"],
                  title="Validation: Actual vs Predicted Daily Demand",
                  labels={"value": "Units Sold", "date": "Date", "variable": "Series"})
st.plotly_chart(fig_val, use_container_width=True)

st.subheader("5. Future Demand Forecast")
forecast = forecast_next_days(daily, model, horizon)
fig_fc = px.line(forecast, x="date", y="forecast_sales", markers=True,
                 title=f"Next {horizon} Days Forecast",
                 labels={"date": "Date", "forecast_sales": "Forecast Units"})
st.plotly_chart(fig_fc, use_container_width=True)

forecast_avg = forecast["forecast_sales"].mean()
forecast_peak = forecast.loc[forecast["forecast_sales"].idxmax()]

st.subheader("6. Business Decision Support")
if forecast_avg > avg_daily_sales:
    st.info("Forecast demand is above the historical average; review inventory and fulfilment capacity.")
else:
    st.info("Forecast demand is below the historical average; review replenishment to avoid unnecessary overstocking.")

st.markdown(f"""
- **Inventory:** adjust replenishment around forecast peaks.
- **Staffing:** review fulfilment capacity before high-demand periods.
- **Promotions:** compare expected demand with planned campaigns.
- **Monitoring:** compare actual demand with predictions and retrain periodically.

**Forecast peak:** {forecast_peak['date'].strftime('%d %b %Y')}  
**Forecast peak demand:** {forecast_peak['forecast_sales']:,.0f} units  
**Forecast average:** {forecast_avg:,.0f} units/day
""")

st.download_button("Download Forecast CSV", forecast.to_csv(index=False).encode("utf-8"),
                   "retail_demand_forecast.csv", "text/csv")

st.caption("Educational prototype; validate forecasts against actual business conditions before operational deployment.")
