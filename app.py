from flask import Flask,render_template,request
import joblib 
import pandas as pd

app=Flask(__name__)
model=joblib.load("salary_model.pkl")
gender_encoder=joblib.load("gender_encoder.pkl")
education_encoder=joblib.load("education_encoder.pkl")
feature_columns=joblib.load("feature_columns.pkl")
job_titles=joblib.load("job_titles.pkl")

genders=dict(enumerate(gender_encoder.classes_))
educations=dict(enumerate(education_encoder.classes_))
jobs = {i:title for i,title in enumerate(job_titles)}

@app.route("/")
def home():
    return render_template(
        "index.html",
        genders=genders,
        educations=educations,
        jobs=jobs,
        age="",
        experience="",
        selected_gender=0,
        selected_education=0,
        selected_job=0
    )
@app.route("/predict", methods=["POST"])
def predict():

    age = float(request.form["age"])
    gender_val = int(request.form["gender"])
    education_val = int(request.form["education"])
    job_val = int(request.form["job"])
    experience = float(request.form["experience"])

    selected_job_title = job_titles[job_val]

    input_data = pd.DataFrame(0, index=[0], columns=feature_columns)

    input_data["Age"] = age
    input_data["Gender"] = gender_val
    input_data["Education Level"] = education_val
    input_data["Years of Experience"] = experience

    job_col_name = f"Job Title_{selected_job_title}"
    if job_col_name in input_data.columns:
        input_data[job_col_name] = 1

    prediction = model.predict(input_data)
    predicted_salary = float(prediction[0])

    monthly_salary = predicted_salary
    yearly_salary = predicted_salary * 12
    if predicted_salary < 50000:
        salary_level = "🟢 Low Salary"
    elif predicted_salary < 100000:
        salary_level = "🟡 Average Salary"
    elif predicted_salary < 150000:
        salary_level = "🔵 High Salary"
    else:
        salary_level = "👑 Excellent Salary"

    r2_score_value = "80.69%"
    mae_value = "₹16,052"
    rmse_value = "₹22,737"

    return render_template(
        "index.html",
        prediction=predicted_salary,
        monthly_salary=monthly_salary,
        yearly_salary=yearly_salary,
        salary_level=salary_level,
        r2_score_value=r2_score_value,
        mae_value=mae_value,
        rmse_value=rmse_value,
        genders=genders,
        educations=educations,
        jobs=jobs,
        age=age,
        experience=experience,
        selected_gender=gender_val,
        selected_education=education_val,
        selected_job=job_val
    )

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)