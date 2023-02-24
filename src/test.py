
import joblib
import os


x = joblib.load (os.path.join(os.path.dirname(__file__), "../models/trained_model.joblib"))

print (x)