import joblib
from numpy import product
import pandas as pd

test_samples = pd.read_csv("Labs/test_samples.csv")
model = joblib.load("Labs/Prediction_model.pkl")

test_samples = test_samples.drop(columns=["BMI_class", "Blood_pressure_class", "Unnamed: 0"], axis=1)

X_test, y_test = test_samples.drop(columns="cardio", axis=1), test_samples["cardio"]
y_pred = model.predict(X_test)

predictions = pd.DataFrame(y_pred)
predictions = predictions.rename({0: "prediction"}, axis=1)
print(predictions)