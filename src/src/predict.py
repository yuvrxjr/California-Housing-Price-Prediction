import joblib
import pandas as pd

model = joblib.load("models/housing_model.pkl")

input_data = pd.read_csv("data/test_data.csv")

predicitons = model.predict(input_data)

input_data["median_house_value"] = predicitons

input_data.to_csv("data/output.csv", index= False)

print("Prediction saved")