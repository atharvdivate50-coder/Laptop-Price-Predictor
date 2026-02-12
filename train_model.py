# train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import pickle

# 1️⃣ Load dataset
df = pd.read_csv("laptop_data.csv")

# 2️⃣ Features and target
categorical_features = ['brand', 'processor_brand', 'processor_series']
numeric_features = ['screen_size_inches', 'ram_gb', 'ssd_gb']

X = df[categorical_features + numeric_features]
y = df['price_inr']

# 3️⃣ One-hot encode categorical features
X = pd.get_dummies(X, columns=categorical_features)

# 4️⃣ Scale numeric features
scaler = StandardScaler()
X[numeric_features] = scaler.fit_transform(X[numeric_features])

# 5️⃣ Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6️⃣ Train model
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# 7️⃣ Save model, scaler, and feature columns
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))
pickle.dump(X.columns.tolist(), open("feature_columns.pkl", "wb"))

print("Model trained and saved successfully!")
