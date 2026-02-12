from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Load model, scaler, and feature columns
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
feature_columns = pickle.load(open("feature_columns.pkl", "rb"))

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input from form
        brand = request.form['brand']
        processor_brand = request.form['processor_brand']
        processor_series = request.form['processor_series']
        screen_size = float(request.form['screen_size_inches'])
        ram = int(request.form['ram_gb'])
        ssd = int(request.form['ssd_gb'])

        # Create DataFrame for input
        input_dict = {
            'screen_size_inches': [screen_size],
            'ram_gb': [ram],
            'ssd_gb': [ssd],
            'brand': [brand],
            'processor_brand': [processor_brand],
            'processor_series': [processor_series]
        }
        input_df = pd.DataFrame(input_dict)

        # One-hot encode categorical features
        input_df = pd.get_dummies(input_df)

        # Add missing columns from training
        for col in feature_columns:
            if col not in input_df.columns:
                input_df[col] = 0

        # Reorder columns to match training
        input_df = input_df[feature_columns]

        # Scale numeric features
        numeric_features = ['screen_size_inches', 'ram_gb', 'ssd_gb']
        input_df[numeric_features] = scaler.transform(input_df[numeric_features])

        # Predict price
        prediction_inr = model.predict(input_df)[0]
        prediction_inr_formatted = "₹ {:,.0f}".format(prediction_inr)

        return render_template("index.html",
                               prediction_text=f"Predicted Price: {prediction_inr_formatted}")

    except Exception as e:
        print(e)  # Shows real error in terminal
        return render_template("index.html",
                               prediction_text="Error in input")

if __name__ == "__main__":
    app.run(debug=True)
