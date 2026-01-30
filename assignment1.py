import pandas as pd
import numpy as np
from statsmodels.tsa.api import VAR

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
# 2. Build VAR dataset
# ----------------------------------------------------
varData = trainData[["trips", "hour"]].astype(float)

# ----------------------------------------------------
# 3. Fit VAR(1)
# ----------------------------------------------------
model = VAR(varData)
modelFit = model.fit(1)   # VAR(1)

# ----------------------------------------------------
# 4. Forecast January (744 hours)
# ----------------------------------------------------
# VAR needs the last k observations as a matrix
last_obs = varData.values[-modelFit.k_ar:]

forecast = modelFit.forecast(last_obs, steps=len(testData))

# Extract only the trips column
pred = forecast[:, 0]