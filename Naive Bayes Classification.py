# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. Load Dataset (Online)
data = load_iris()
X = data.data
y = data.target
feature_names = data.feature_names
target_names = data.target_names

# 2. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Create Model
model = GaussianNB()

# 4. Train Model
model.fit(X_train, y_train)

# Get class names
class_names = data.target_names
print(class_names)

for i in range(3):
    plt.scatter(X[y == i, 0], X[y == i, 2], label=f"Class {i}")

plt.xlabel("Sepal Length")
plt.ylabel("Petal Length")
plt.title("Iris Dataset Visualization")
plt.legend()
plt.show()

# 5. Predict
y_pred = model.predict(X_test)

# 6. Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy * 100:.2f}%")

# 7. Classification Report
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# 8. Display Some Predictions
print("\nSample Predictions:")
for i in range(5):
    print(f"Predicted: {y_pred[i]}, Actual: {y_test[i]}")