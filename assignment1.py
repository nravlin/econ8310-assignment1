import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

# ----------------------------------------------------
# 1. Load data
# ----------------------------------------------------
trainData = pd.read_csv("assignment_data_train.csv")
testData  = pd.read_csv("assignment_data_test.csv")

trainData["Timestamp"] = pd.to_datetime(trainData["Timestamp"])
testData["Timestamp"]  = pd.to_datetime(testData["Timestamp"])

trainData = trainData.sort_values("Timestamp").set_index("Timestamp")
testData  = testData.sort_values("Timestamp").set_index("Timestamp")

# ----------------------------------------------------
# 2. Endogenous and exogenous variables
# ----------------------------------------------------
y = trainData["trips"].astype(float)

# Exogenous regressors: hour, month, day
train_exog = trainData[["hour", "month", "day"]].astype(float)
test_exog  = testData[["hour", "month", "day"]].astype(float)

# ----------------------------------------------------
# 3. Define SARIMAX model
# ----------------------------------------------------
# ARMA(1,1) with exogenous regressors
model = SARIMAX(
    y,
    exog=train_exog,
    order=(1, 0, 1),
    trend="c",
    enforce_stationarity=False,
    enforce_invertibility=False
)

# ----------------------------------------------------
# 4. Fit model
# ----------------------------------------------------
modelFit = model.fit(disp=False)

# ----------------------------------------------------
# 5. Forecast January
# ----------------------------------------------------
pred = modelFit.forecast(
    steps=len(testData),
    exog=test_exog
)

pred = np.array(pred)