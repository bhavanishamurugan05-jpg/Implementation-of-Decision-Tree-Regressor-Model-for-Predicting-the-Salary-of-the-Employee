# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries and load the employee salary dataset
2. Separate the input features and target variable, then split the dataset into training and testing sets.
3. Create the Decision Tree Regressor model and train it using the training data.
4.Predict the employee salary using the test data and evaluate the model performance. 

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: BHAVANISHA.M
RegisterNumber:  212225230034
*/
```
```
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import accuracy_score, classification_report
data = pd.read_csv("Salary.csv")
X = data.drop("Salary", axis=1)
y = data["Salary"]
X = pd.get_dummies(X, drop_first=True)
model = DecisionTreeRegressor(random_state=42)
model.fit(X, y)
y_pred = model.predict(X)
print("\nAccuracy:", accuracy_score(y, y_pred)*100)
print("\nClassification Report:")
print(classification_report(y, y_pred))
plt.figure(figsize=(25,12))
plot_tree(
    model,
    feature_names=X.columns,
    filled=True
)
plt.title("Decision Tree Regressor")
plt.show()
```
## Output:
<img width="941" height="472" alt="image" src="https://github.com/user-attachments/assets/8d3c8b8e-7e68-4152-8bc1-ee9ded671de2" />

<img width="1567" height="712" alt="image" src="https://github.com/user-attachments/assets/ffa09b22-171b-4500-9987-7beffbe5b9c1" />


## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
