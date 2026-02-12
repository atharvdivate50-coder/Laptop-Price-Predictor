import pickle
import numpy as np

# Load your model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Print the number of features the scaler expects
print("Scaler expects this many features:", len(scaler.scale_))

# Optional: print scale values to see more details
print("Scaler scale values:", scaler.scale_)

# Test prediction with only numeric features (6 features)
try:
    test_features_numeric = np.array([[15.6, 2.5, 8, 2.0, 256, 1000]])
    test_scaled = scaler.transform(test_features_numeric)
    print("Prediction with numeric features only:", model.predict(test_scaled))
    print("Your model uses numeric features only.")
except Exception as e:
    print("Error:", e)
    print("Your model likely requires more features (e.g., brand features).")
