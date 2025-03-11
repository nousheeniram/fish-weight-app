from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained ML model
model = pickle.load(open("fish_weight_model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")  # Load the HTML page

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get input values from the form
        species = int(request.form["species"])
        length1 = float(request.form["length1"])
        length2 = float(request.form["length2"])
        length3 = float(request.form["length3"])
        height = float(request.form["height"])
        width = float(request.form["width"])

        # Create input array for prediction
        input_features = np.array([[species, length1, length2, length3, height, width]])

        # Make prediction
        prediction = model.predict(input_features.reshape(1, -1))[0]

        # Return prediction
        return render_template("index.html", prediction_text=f"Predicted Fish Weight: {prediction:.2f} grams")

    except Exception as e:
        return str(e)

if __name__ == "__main__":
    app.run(debug=True)
