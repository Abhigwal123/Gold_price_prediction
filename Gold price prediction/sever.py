from flask import Flask, render_template, request
import pickle
import joblib
app = Flask(__name__)

# Load the trained model
model = joblib.load('/regressor_model')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input values from the form
    SPX = float(request.form['SPX'])
    USO = float(request.form['USO'])
    SLV = float(request.form['SLV'])
    EURUSD = float(request.form['EURUSD'])

    # Perform prediction using the loaded model
    prediction = model.predict([[SPX, USO, SLV, EURUSD]])

    # Render the result.html page with the predicted result
    return render_template('result.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run()
