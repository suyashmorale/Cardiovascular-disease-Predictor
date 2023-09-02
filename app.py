from flask import Flask, render_template, request
import pickle
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the pre-trained machine learning model
model_path = "model/classifier.pkl"
with open(model_path, "rb") as file:
    classifier = pickle.load(file)

# Load the StandardScaler used during training
scaler_path = "model/scaler.pkl"
with open(scaler_path, "rb") as file:
    scaler = pickle.load(file)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        age = int(request.form["age"])
        gender = 0 if request.form["gender"] == "Male" else 1
        height = int(request.form["height"])
        weight = int(request.form["weight"])
        ap_hi = int(request.form["ap_hi"])
        ap_lo = int(request.form["ap_lo"])

        cholesterol = 1  # Default value for 'Normal'
        if request.form["cholesterol"] == "Above normal":
            cholesterol = 2
        elif request.form["cholesterol"] == "Well above normal":
            cholesterol = 3

        gluc = 1  # Default value for 'Normal'
        if request.form["gluc"] == "Above normal":
            gluc = 2
        elif request.form["gluc"] == "Well above normal":
            gluc = 3

        smoke = 1 if request.form["smoke"] == "Yes" else 0
        alco = 1 if request.form["alco"] == "Yes" else 0
        active = 1 if request.form["active"] == "Yes" else 0

        # Scale the input data before making predictions
        scaled_input = scaler.transform([[age, gender, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active]])

        prediction = predict_note_authentication(scaled_input)

        return render_template("result.html", prediction=prediction)

    return render_template("index.html")


def predict_note_authentication(input_data):
    prediction = classifier.predict(input_data)
    return prediction[0]

if __name__ == "__main__":
    app.run(debug=True)
