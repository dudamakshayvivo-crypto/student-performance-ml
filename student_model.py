import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Dataset (study hours vs marks)
hours = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]).reshape(-1, 1)
marks = np.array([15, 18, 25, 35, 45, 55, 65, 75, 85])

# Split data
X_train, X_test, y_train, y_test = train_test_split(hours, marks, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Accuracy
score = model.score(X_test, y_test)
print("Model Accuracy:", round(score, 2))

# Predict for 5 hours
user_hours = float(input("Enter study hours: "))
prediction = model.predict([[user_hours]])

print("Predicted Marks:", round(prediction[0], 2))

# Graph
plt.scatter(hours, marks)
plt.plot(hours, model.predict(hours))
plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.title("Student Performance Prediction")
plt.show()
