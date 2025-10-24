# app.py
from flask import Flask, request, render_template
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

app = Flask(__name__)

# ---------------- Load and preprocess data ----------------
df = pd.read_csv("Heart_Disease_and_Hospitals.csv")
df = df.drop(['country','treatment_date','first_name', 'last_name','full_name'], axis=1)

encoder = LabelEncoder()
df['gender'] = encoder.fit_transform(df['gender'])

# ---------------- Load the saved model ----------------
with open("model.pkl", "rb") as f:
    xgb = pickle.load(f)

# Define the feature columns
feature_columns = ['age','blood_pressure','cholesterol','bmi','glucose_level','gender']
X = df[feature_columns]
y = df['heart_disease']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------- Train XGBoost model ----------------
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb.fit(X_train, y_train)

# ---------------- Flask Routes ----------------
@app.route('/')
def home():
    return render_template('index.html')

# Single Entry Prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        age = float(request.form['age'])
        blood_pressure = float(request.form['blood_pressure'])
        cholesterol = float(request.form['cholesterol'])
        bmi = float(request.form['bmi'])
        glucose_level = float(request.form['glucose_level'])
        gender = int(request.form['gender'])  # 1=Male, 0=Female

        # Create dataframe for prediction
        input_data = pd.DataFrame([[age, blood_pressure, cholesterol, bmi, glucose_level, gender]],
                                  columns=feature_columns)

        # Predict
        prediction = xgb.predict(input_data)[0]
        result = "Heart Disease Detected" if prediction == 1 else "No Heart Disease"

        return render_template('index.html', prediction_text=result)
    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

# Batch Prediction via CSV
@app.route('/predict_file', methods=['POST'])
def predict_file():
    try:
        file = request.files['file']
        if not file:
            return render_template('index.html', prediction_text="No file uploaded")

        df_file = pd.read_csv(file)

        # Check required columns
        for col in feature_columns:
            if col not in df_file.columns:
                return render_template('index.html', prediction_text=f"Missing column: {col}")

        # Ensure gender column is numeric
        df_file['gender'] = df_file['gender'].astype(int)

        # Predict
        predictions = xgb.predict(df_file[feature_columns])
        df_file['Prediction'] = ['Heart Disease' if p==1 else 'No Heart Disease' for p in predictions]

        # Convert to HTML table
        prediction_html = df_file.to_html(classes='data', index=False)

        return render_template('index.html', prediction_text=prediction_html)
    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
