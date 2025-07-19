import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib

# Set the path where your image folders are located
base_path = "C:/Users/Dell/Downloads/archive/poultry_diseases"

# These folder names are your class labels
classes = ["ncd", "salmo", "healthy", "cocci"]

X = []
y = []

# Load and process each image
print("🔄 Loading images and extracting features...")
for label in classes:
    folder_path = os.path.join(base_path, label)
    if not os.path.exists(folder_path):
        print(f"⚠️ Folder not found: {folder_path}")
        continue
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".jpg"):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (100, 100))  # Resize image to 100x100
                img_flatten = img.flatten()  # Convert to 1D array
                X.append(img_flatten)
                y.append(label)

print(f"✅ Total images loaded: {len(X)}")

# Convert to numpy arrays
X = np.array(X)
y = np.array(y)

# Split data for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Decision Tree model
print("🌲 Training Decision Tree model...")
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"🎯 Decision Tree Accuracy: {accuracy * 100:.2f}%")

# Save the trained model
model_path = "decision_tree_model.pkl"
joblib.dump(model, model_path)
print(f"💾 Model saved to: {model_path}")
