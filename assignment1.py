import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# ----------------------------------------------------
# 1. Load training and test data
# ----------------------------------------------------
trainData = pd.read_csv("assignment_data_train.csv")
testData  = pd.read_csv("assignment_data_test.csv")

# ----------------------------------------------------
# 2. Parse timestamps and sort
# ----------------------------------------------------
trainData["Timestamp"] = pd.to_datetime(trainData["Timestamp"])
testData["Timestamp"]  = pd.to_datetime(testData["Timestamp"])

trainData = trainData.sort_values("Timestamp").set_index("Timestamp")
testData  = testData.sort_values("Timestamp").set_index("Timestamp")

# ----------------------------------------------------
# 3. Define dependent variable
# ----------------------------------------------------
trainTrips = trainData["trips"]

# ----------------------------------------------------
# 4. Define the model (algorithm)
# ----------------------------------------------------
model = ExponentialSmoothing(
    trainTrips,
    trend="add",
    seasonal="add",
    seasonal_periods=24
)

# ----------------------------------------------------
# 5. Fit the model
# ----------------------------------------------------
modelFit = model.fit(optimized=True)

# ----------------------------------------------------
# 6. Forecast for the test period (January)
# ----------------------------------------------------
forecast_horizon = len(testData)
pred = np.array(modelFit.forecast(forecast_horizon))