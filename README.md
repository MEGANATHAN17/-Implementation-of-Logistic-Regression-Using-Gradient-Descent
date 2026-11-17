# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. Import the required libraries, load the placement dataset using Pandas, create a copy of the dataset, and remove unnecessary columns (`sl_no`, `salary`).

2. Convert all categorical values such as gender, board type, work experience, and specialization into numerical form using `LabelEncoder`.

3. Separate the input features (`x`) and target variable (`y`), define the sigmoid function, loss function, and apply gradient descent to train the logistic regression model.

4. Predict placement status using the trained model, calculate accuracy by comparing predicted and actual values, and test the model with new input data.

## Program:
```
/*
Developed by: MEGANATHAN R
RegisterNumber:  212224230156
*/

import numpy as np
import pandas as pd
from google.colab import files
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Upload Dataset
uploaded = files.upload()
filename = next(iter(uploaded))

# Load Dataset
data = pd.read_csv(filename)

print("First 5 rows:")
print(data.head())

# Data Preprocessing
data = data.dropna()

# Encode Categorical Columns
le = LabelEncoder()

for col in data.select_dtypes(include=['object']).columns:
    data[col] = le.fit_transform(data[col].astype(str))

# Features and Target
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Convert Target to Binary
y = LabelEncoder().fit_transform(y)
y = y.reshape(-1, 1)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Add Bias Column
X_train = np.c_[np.ones((len(X_train), 1)), X_train]
X_test = np.c_[np.ones((len(X_test), 1)), X_test]

# Sigmoid Function
def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

# Logistic Regression Using Gradient Descent
def logistic_regression(X, y, learning_rate=0.01, num_iters=1000):

    theta = np.zeros((X.shape[1], 1))

    for i in range(num_iters):
        z = X.dot(theta)
        predictions = sigmoid(z)

        errors = predictions - y
        gradients = (1 / len(y)) * X.T.dot(errors)

        theta = theta - learning_rate * gradients

    return theta

# Train Model
theta = logistic_regression(X_train, y_train)

print("\nModel Parameters:")
print(theta)

# Prediction
y_prob = sigmoid(X_test.dot(theta))
y_pred = (y_prob >= 0.5).astype(int)

# Model Evaluation
print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# Display Actual and Predicted Values
results = pd.DataFrame({
    "Actual": y_test.flatten(),
    "Predicted": y_pred.flatten()
})

print("\nActual vs Predicted:")
print(results)

```

## Output:


<img width="893" height="547" alt="image" src="https://github.com/user-attachments/assets/b33bb1f7-8782-4296-8e9b-d3ef5d16fc30" />

<img width="506" height="402" alt="image" src="https://github.com/user-attachments/assets/19412c62-9709-4377-9351-a867fb087d00" />


<img width="346" height="512" alt="image" src="https://github.com/user-attachments/assets/05f84291-8341-4037-94c0-7766502f974e" />

## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

