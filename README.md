import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
DATA_DIR = "path_to_dataset"
CATEGORIES = ["Dog", "Cat"]
IMG_SIZE = 64  # Resize images to 64x64

def load_data():
    data = []
    labels = []
    for category in CATEGORIES:
        path = os.path.join(DATA_DIR, category)
        label = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                data.append(img_array.flatten())  # Flatten image
                labels.append(label)
            except Exception as e:
                print(f"Error loading image {img}: {e}")
    return np.array(data), np.array(labels)

# Prepare dataset
X, y = load_data()
X = X / 255.0  # Normalize pixel values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train SVM model
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)

# Predictions
y_pred = svm_model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
print(classification_report(y_test, y_pred))
