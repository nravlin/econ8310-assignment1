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
# trips + hour (float)
endog = trainData[["trips", "hour"]].astype(float)

# ----------------------------------------------------
# 3. Exogenous variables
# ----------------------------------------------------
# month + day + a *tiny* interaction term (keeps it original)
train_exog = trainData[["month", "day"]].astype(float)
train_exog["md"] = train_exog["month"] * 0.01 + train_exog["day"] * 0.001

test_exog = testData[["month", "day"]].astype(float)
test_exog["md"] = test_exog["month"] * 0.01 + test_exog["day"] * 0.001

# ----------------------------------------------------
# 4. Define model
# ----------------------------------------------------
# VARMAX(1,1) with constant trend
model = VARMAX(
    endog=endog,
    exog=train_exog,
    order=(1, 1),
    trend="c"
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

pred = forecast["trips"].values