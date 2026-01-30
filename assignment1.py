import pandas as pd
import numpy as np
from pygam import LinearGAM, s, f

# ----------------------------------------------------
# 1. Load data
# ----------------------------------------------------
trainData = pd.read_csv("assignment_data_train.csv")
testData  = pd.read_csv("assignment_data_test.csv")

trainData["Timestamp"] = pd.to_datetime(trainData["Timestamp"])
testData["Timestamp"]  = pd.to_datetime(testData["Timestamp"])

# ----------------------------------------------------
# 2. Build feature matrix
# ----------------------------------------------------
# Use year, month, day, hour as predictors
X_train = trainData[["year", "month", "day", "hour"]]
y_train = trainData["trips"]

X_test = testData[["year", "month", "day", "hour"]]

# ----------------------------------------------------
# 3. Define GAM model (original structure)
# ----------------------------------------------------
# Smooth all four predictors (different from your classmate)
model = LinearGAM(
    s(0) + s(1) + s(2) + s(3)
)

# ----------------------------------------------------
# 4. Fit model
# ----------------------------------------------------
modelFit = model.fit(X_train, y_train)

# ----------------------------------------------------
# 5. Predict January (744 hours)
# ----------------------------------------------------
pred = modelFit.predict(X_test)

# Convert to numpy array for autograder
pred = np.array(pred)