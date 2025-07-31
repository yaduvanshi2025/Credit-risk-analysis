from flask import Flask, render_template, request
import joblib
import numpy as np

# Create Flask app
app = Flask(__name__)

# Load your saved model
model = joblib.load('credit_model.pkl')  # Make sure this is the correct filename

# Homepage route
@app.route('/')
def home():
    return render_template('index.html')  # This should be in the 'templates' folder

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get input data from form
        age = float(request.form['age'])
        income = float(request.form['income'])
        employ = float(request.form['employ'])
        address = float(request.form['address'])
        ed = float(request.form['ed'])
        debtinc = float(request.form['debtinc'])
        creddebt = float(request.form['creddebt'])
        othdebt = float(request.form['othdebt'])

        # Combine inputs into array
        input_data = np.array([[age, ed, employ, address, income, debtinc, creddebt, othdebt]])

        # Predict
        prediction = model.predict(input_data)

        result = 'Default' if prediction[0] == 1 else 'No Default'
        return render_template('index.html', prediction=result)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)