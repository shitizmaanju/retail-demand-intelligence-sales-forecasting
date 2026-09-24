# Retail Demand Intelligence & Sales Forecasting

## Project Overview
A retail analytics dashboard that converts historical store-item sales into trends, forecasting results, and business decision support.

**Business Intelligence flow:** Data → Information → Insights → Risks/Opportunities → Action

## Business Problem
How can historical store-item sales data be used to understand demand patterns and forecast near-term retail demand so that inventory and operational decisions can be better planned?

## Dataset
**Store Item Demand Forecasting Challenge**

Dataset: https://www.kaggle.com/competitions/demand-forecasting-kernels-only/data

The dataset contains five years of daily sales for 50 items across 10 stores. Main fields:
- `date`
- `store`
- `item`
- `sales`

## Analytics
- Total units sold
- Average daily demand
- Peak daily demand
- Store performance
- Item performance
- Daily demand trend
- Short-term demand forecast
- Forecast validation using MAE and RMSE

## Forecasting Model
Random Forest Regression with:
- calendar features
- 1/7/14/28-day lag features
- 7/28-day rolling demand features

An 80/20 chronological train-validation split is used.

## Technology
Python, Pandas, NumPy, Scikit-learn, Plotly, Streamlit

## Project Files
```text
app.py
requirements.txt
README.md
PROJECT_REPORT.docx
```

## Quick Start
1. Download `train.csv` from the dataset link above.
2. Put `train.csv` beside `app.py`.
3. Install dependencies:
```bash
pip install -r requirements.txt
```
4. Start the dashboard:
```bash
streamlit run app.py
```

## Business Decision Support
The dashboard helps review:
- inventory/replenishment needs
- fulfilment and staffing capacity
- promotion timing
- forecast error and monitoring

## Limitations
The current model forecasts aggregate daily demand. Future versions can add store-item forecasts, promotions, holidays, prediction intervals, safety-stock calculations, and automated retraining.

## Reproducibility
The raw dataset is not included in this repository. Obtain it from the original source linked above.
