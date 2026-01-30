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

# ----------------------------------------------------
# 2. Build endogenous and exogenous matrices
# ----------------------------------------------------
# Endogenous: trips + hour (float required for VARMAX)
endog = trainData[["trips", "hour"]].astype("float64")

# Exogenous: month, day, and interaction term
train_exog = trainData[["month", "day"]].copy()
train_exog["md_interaction"] = train_exog["month"] * train_exog["day"]
train_exog = train_exog.astype("float64")

test_exog = testData[["month", "day"]].copy()
test_exog["md_interaction"] = test_exog["month"] * test_exog["day"]
test_exog = test_exog.astype("float64")

# ----------------------------------------------------
# 3. Define model
# ----------------------------------------------------
# VARMAX(1,0) is stable and avoids dtype crashes
model = VARMAX(
    endog=endog,
    exog=train_exog,
    order=(1, 0),
    trend="c"
)

# ----------------------------------------------------
# 4. Fit model
# ----------------------------------------------------
modelFit = model.fit(disp=False)

# ----------------------------------------------------
# 5. Forecast January (744 hours)
# ----------------------------------------------------
forecast = modelFit.forecast(
    steps=len(testData),
    exog=test_exog
)

# Extract only the trips forecast
pred = forecast["trips"].values