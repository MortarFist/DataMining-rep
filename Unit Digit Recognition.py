import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras import models, layers

# 1. Load Dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 2. Normalize Data (0–255 → 0–1)
x_train = x_train / 255.0
x_test = x_test / 255.0

# 3. Build Model
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# 4. Compile Model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 5. Train Model
model.fit(x_train, y_train, epochs=5)

# 6. Evaluate Model
loss, accuracy = model.evaluate(x_test, y_test)
print(f"\nTest Accuracy: {accuracy * 100:.2f}%")

# 7. Make Predictions
predictions = model.predict(x_test)

# 8. Display Results (IMPORTANT FOR ASSIGNMENT)
plt.figure(figsize=(10, 5))

for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_test[i], cmap='gray')
    
    predicted_label = np.argmax(predictions[i])
    actual_label = y_test[i]
    
    plt.title(f"P: {predicted_label} / A: {actual_label}")
    plt.axis('off')

plt.tight_layout()
plt.show()