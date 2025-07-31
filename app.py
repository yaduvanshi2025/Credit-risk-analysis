from flask import Flask, render_template, request, redirect, url_for, session
import joblib
import numpy as np

app = Flask(__name__)
app.secret_key = '99a1d20e197bd03a7c271b3ac2f50a24'


# Model load
model = joblib.load('credit_model.pkl')

@app.route('/')
def home():
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def login():
    name = request.form['name']
    email = request.form['email']
    password = request.form['password']

    # Dummy: save in session
    session['name'] = name
    session['email'] = email

    return redirect(url_for('dashboard'))

@app.route('/dashboard')
def dashboard():
    if 'name' not in session:
        return redirect(url_for('home'))
    return render_template('dashboard.html', name=session['name'])

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if 'name' not in session:
        return redirect(url_for('home'))

    if request.method == 'POST':
        try:
            age = float(request.form['age'])
            ed = float(request.form['ed'])
            employ = float(request.form['employ'])
            address = float(request.form['address'])
            income = float(request.form['income'])
            debtinc = float(request.form['debtinc'])
            creddebt = float(request.form['creddebt'])
            othdebt = float(request.form['othdebt'])

            input_data = np.array([[age, ed, employ, address, income, debtinc, creddebt, othdebt]])
            prediction = model.predict(input_data)
            predicted_class = int(prediction[0])
            probability = model.predict_proba(input_data)
            confidence = round(probability[0][predicted_class] * 100, 2)
            result = 'Default' if predicted_class == 1 else 'No Default'

            if predicted_class == 1:
                if confidence >= 70:
                    risk_category = 'High Risk'
                elif confidence >= 40:
                    risk_category = 'Medium Risk'
                else:
                    risk_category = 'Low Risk'
            else:
                if confidence >= 70:
                    risk_category = 'Low Risk'
                elif confidence >= 40:
                    risk_category = 'Medium Risk'
                else:
                    risk_category = 'High Risk'

            intercept = round(float(model.intercept_[0]), 4)
            coefficients = [round(c, 4) for c in model.coef_[0]]

            return render_template(
                'predict.html',
                prediction=result,
                confidence=confidence,
                risk_category=risk_category,
                intercept=intercept,
                coefficients=coefficients
            )

        except Exception as e:
            return f"Error: {e}"

    return render_template('predict.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)
