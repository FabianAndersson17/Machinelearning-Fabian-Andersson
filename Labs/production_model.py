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
print(predict_probability) 
print(y_pred)
predictions = pd.DataFrame([y_pred, predict_probability]).T
predictions = predictions.rename({0: "prediction", 1: "Probability"}, axis=1)
print(predictions)
## Gives a list of lists with 1 and 0. This happens because the decision tree can either go "left" or "rigth" 
## Dependent on if the values of the data point are according to the parameters of the left path or right path
## https://stackoverflow.com/questions/47251594/scikit-learn-decision-tree-probability-of-prediction-being-a-or-b?msclkid=77315ed3ab8211ec9f236be39ea50843
## https://towardsdatascience.com/predict-vs-predict-proba-scikit-learn-bdc45daa5972#:~:text=The%20predict_proba%20%28%29%20method%20In%20the%20context%20of,returns%20the%20class%20probabilities%20for%20each%20data%20point.?msclkid=1fd6a50bab8211ecbbf44f281ea62deb
