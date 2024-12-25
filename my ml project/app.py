from flask import Flask, render_template, request
import pickle
import numpy as np

# Initialize the Flask app
app = Flask(__name__)

# Load the model with error handling
try:
    model = pickle.load(open('model.pkl', 'rb'))
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=["POST"])
def predict():
    if not model:
        return render_template('index.html', prediction_text='Error: Model not loaded.')

    try:
        # Parse input features from form
        feature = [int(x) for x in request.form.values()]
        feature_final = np.array(feature).reshape(1, -1)  # Ensure it's shaped for the model
        prediction = model.predict(feature_final)
        return render_template(
            'index.html',
            prediction_text=f'Price of House will be Rs. {int(prediction[0])}'
        )
    except Exception as e:
        return render_template('index.html', prediction_text=f"Error during prediction: {e}")

if __name__ == '__main__':
    app.run(debug=True)
