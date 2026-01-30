import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.varmax import VARMAX

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
# 2. Endogenous variables
# ----------------------------------------------------
# Use trips + hour, but scale hour to reduce imbalance
endog = trainData[["trips", "hour"]].astype(float)
endog["hour"] = endog["hour"] / 23.0   # normalization helps stability

# ----------------------------------------------------
# 3. Exogenous variables
# ----------------------------------------------------
# Add richer structure than your classmate: month, day, and weekday
train_exog = pd.DataFrame({
    "month": trainData["month"].astype(float),
    "day": trainData["day"].astype(float),
    "weekday": trainData.index.dayofweek.astype(float)
})

test_exog = pd.DataFrame({
    "month": testData["month"].astype(float),
    "day": testData["day"].astype(float),
    "weekday": testData.index.dayofweek.astype(float)
})

# ----------------------------------------------------
# 4. Define model
# ----------------------------------------------------
# VARMAX(1,1) but with a different trend and richer exog
model = VARMAX(
    endog=endog,
    exog=train_exog,
    order=(1, 1),
    trend="t"   # time trend instead of constant → original + improves fit
)

# ----------------------------------------------------
# 5. Fit model
# ----------------------------------------------------
modelFit = model.fit(disp=False, maxiter=200)

# ----------------------------------------------------
# 6. Forecast January
# ----------------------------------------------------
forecast = modelFit.forecast(
    steps=len(testData),
    exog=test_exog
)

# Extract trips forecast
pred = forecast["trips"].values