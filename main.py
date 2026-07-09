from venv import create

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

#preprocessing data

from sklearn.preprocessing import LabelEncoder

# Gender Encoder
gender_encoder = LabelEncoder()
data["Gender"] = gender_encoder.fit_transform(data["Gender"])

# Education Encoder
education_encoder = LabelEncoder()
data["Education Level"] = education_encoder.fit_transform(data["Education Level"])

# Job Title Encoder
job_encoder = LabelEncoder()
data["Job Title"] = job_encoder.fit_transform(data["Job Title"])
print(data.head())

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
from sklearn.metrics import r2_score

score = r2_score(y_test, y_pred)

print(score)

#mean absolute error
from sklearn.metrics import mean_absolute_error

print(mean_absolute_error(y_test, y_pred))

#mean squared error
from sklearn.metrics import mean_squared_error

print(mean_squared_error(y_test, y_pred))

#root mean squared error
import numpy as np

rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(rmse)

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
joblib.dump(job_encoder, "job_encoder.pkl")

print("Model & Encoders Saved Successfully")

print("Model Saved Successfully")
print(data["Education Level"].unique())
print(data["Job Title"].unique())






