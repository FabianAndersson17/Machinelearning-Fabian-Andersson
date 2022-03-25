from time import process_time_ns
import joblib
from numpy import product
import pandas as pd

test_samples = pd.read_csv("Labs/test_samples.csv")
model = joblib.load("Labs/Prediction_model.pkl")

test_samples = test_samples.drop(columns=["BMI_class", "Blood_pressure_class", "Unnamed: 0"], axis=1)

X_test, y_test = test_samples.drop(columns="cardio", axis=1), test_samples["cardio"]
y_pred = model.predict(X_test)

predict_probability = model.predict_proba(X_test)

probability_0 = [i[0] for i in predict_probability] ## Code taken from: https://stackoverflow.com/questions/25050311/extract-first-item-of-each-sublist
probability_1 = [i[1] for i in predict_probability]

print(predict_probability) 
print(y_pred)
predictions = pd.DataFrame([y_pred,probability_0 ,probability_1]).T
predictions = predictions.rename({0: "prediction", 1: "Probability for 0", 2: "Probability for 1"}, axis=1)
predictions.to_csv("predictions.csv")
print(predictions)
