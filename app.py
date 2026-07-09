from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load trained model
model = joblib.load("salary_model.pkl")

# Encoded values (same as training)
genders = {
    0: "Female",
    1: "Male"
}

educations = {
    0: "Education 0",
    1: "Education 1",
    2: "Education 2",
    3: "Education 3"
}

jobs = {
    0: "Job 0",
    1: "Job 1",
    2: "Job 2",
    3: "Job 3",
    4: "Job 4"
}


@app.route("/")
def home():
    return render_template(
        "index.html",
        genders=genders,
        educations=educations,
        jobs=jobs
    )


@app.route("/predict", methods=["POST"])
def predict():

    age = float(request.form["age"])
    gender = int(request.form["gender"])
    education = int(request.form["education"])
    job = int(request.form["job"])
    experience = float(request.form["experience"])

    prediction = model.predict([[age, gender, education, job, experience]])

    return render_template(
        "index.html",
        prediction=f"Predicted Salary : ₹ {prediction[0]:,.2f}",
        genders=genders,
        educations=educations,
        jobs=jobs
    )


if __name__ == "__main__":
    app.run(debug=True)