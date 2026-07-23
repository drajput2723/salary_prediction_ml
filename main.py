import pandas as pd
import numpy as np

#dataset load
data = pd.read_csv("Clean_Salary_Data.csv")
#First 5 rows
print(data.head())
#last 5 rows
print(data.tail())
#rows and columns
print(data.shape)
#information about dataset
print(data.info())
#statitics
print(data.describe())
#columns
print(data.columns)
#missing values
print(data.isnull().sum())
#duplicate rows
print(data.duplicated().sum())
#highest salary
print(data["Salary"].max())
#lowest salary
print(data["Salary"].min())
#average salary
print(data["Salary"].mean())

#cleaning data
data = data.dropna()
data = data.drop_duplicates()
#check again for missing values and duplicates
print(data.shape)
print(data.isnull().sum())
print(data.duplicated().sum())
#save cleaned data
data.to_csv("Clean_Salary_Data.csv", index=False)
print(sorted(data["Education Level"].unique()))
print(sorted(data["Job Title"].unique()))

# Education Level cleaning
data["Education Level"] = data["Education Level"].str.strip().replace({
    "Bachelor's Degree": "Bachelor's",
    "Master's Degree": "Master's",
    "phD": "PhD"
})

# check - ab sirf 4 unique values honi chahiye
print(sorted(data["Education Level"].unique()))

job_counts = data["Job Title"].value_counts()
rare_jobs = job_counts[job_counts < 10].index
data["Job Title"] = data["Job Title"].replace(rare_jobs, "Other")
print(data["Job Title"].nunique())  # ab kaafi kam categories honi chahiye

#preprocessing data

from sklearn.preprocessing import LabelEncoder

# Gender Encoder
gender_encoder = LabelEncoder()
data["Gender"] = gender_encoder.fit_transform(data["Gender"])

# Education Encoder
education_encoder = LabelEncoder()
data["Education Level"] = education_encoder.fit_transform(data["Education Level"])

# Job Title - One-Hot Encoding
data = pd.get_dummies(data, columns=["Job Title"], drop_first=True)

print(data.head())
print(data.shape)

#feature selection
X = data.drop("Salary", axis=1)
y = data["Salary"]

#train_test_split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1
)

print(X_train.head())
print(X_test.head())

print(X_train.shape)
print(X_test.shape)

#model import
from sklearn.linear_model import LinearRegression
#model create
model = LinearRegression()
#model train
model.fit(X_train, y_train)
#model predict
y_pred = model.predict(X_test)

#actual vs predicted
import pandas as pd

comparison = pd.DataFrame({
    "Actual": y_test,
    "Predicted": y_pred
})

print(comparison.head())

#accuracy/evaluation
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("\n========== MODEL EVALUATION ==========")
print(f"R² Score  : {r2:.4f}")
print(f"MAE       : {mae:.2f}")
print(f"MSE       : {mse:.2f}")
print(f"RMSE      : {rmse:.2f}")
print("======================================")

#graphs
import matplotlib.pyplot as plt

plt.scatter(y_test, y_pred)

plt.xlabel("Actual Salary")
plt.ylabel("Predicted Salary")
plt.title("Actual vs Predicted Salary")

plt.show()

#save model
import joblib

joblib.dump(model, "salary_model.pkl")
joblib.dump(gender_encoder, "gender_encoder.pkl")
joblib.dump(education_encoder, "education_encoder.pkl")


print("Model & Encoders Saved Successfully")


# Save feature column order (needed for app.py to build correct input)
feature_columns = X.columns.tolist()
joblib.dump(feature_columns, "feature_columns.pkl")

# Save list of job titles (after "Other" grouping) for dropdown in app.py
job_titles_list = sorted(data["Job Title"].unique().tolist()) if "Job Title" in data.columns else None
# Since Job Title is now one-hot encoded, get list from original column names
job_title_columns = [col.replace("Job Title_", "") for col in X.columns if col.startswith("Job Title_")]
joblib.dump(job_title_columns, "job_titles.pkl")

print("Feature columns & job titles saved successfully")

print("Model Saved Successfully")
print(data["Education Level"].unique())






