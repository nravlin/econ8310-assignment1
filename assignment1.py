import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing

trainData = pd.read_csv("assignment_data_train.csv")
testData  = pd.read_csv("assignment_data_test.csv")

trainData["Timestamp"] = pd.to_datetime(trainData["Timestamp"])
testData["Timestamp"]  = pd.to_datetime(testData["Timestamp"])

trainData = trainData.sort_values("Timestamp").set_index("Timestamp")
testData  = testData.sort_values("Timestamp").set_index("Timestamp")

trainTrips = trainData["trips"]

# Weekly seasonality (24 hours × 7 days)
model = ExponentialSmoothing(
    trainTrips,
    trend="add",
    seasonal="add",
    seasonal_periods=168
)

modelFit = model.fit(optimized=True)

forecast_horizon = len(testData)
pred = np.array(modelFit.forecast(forecast_horizon))