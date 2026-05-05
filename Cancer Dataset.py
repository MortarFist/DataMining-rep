# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. Load Dataset
data = load_breast_cancer()
X = data.data
y = data.target
target_names = data.target_names  # malignant, benign

# 2. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Get class names
class_names = data.target_names
print(class_names)

# 3. Create Model (KNN)
model = KNeighborsClassifier(n_neighbors=5)

# 4. Train Model
model.fit(X_train, y_train)

# 5. Predict
y_pred = model.predict(X_test)

# 6. Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy * 100:.2f}%")

# 7. Classification Report
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=target_names))

# 8. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure()
sns.heatmap(cm, annot=True, fmt='d')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# 9. Sample Predictions
print("\nSample Predictions:")
for i in range(5):
    print(f"Predicted: {target_names[y_pred[i]]}, Actual: {target_names[y_test[i]]}")