import os
import pandas as pd
import numpy as np
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing import image
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold

# Load data from Excel sheets
print("Loading data from Excel sheets...")
healthy_data = pd.read_excel('/home/labadmin/R7A_group11/augmented dataset/combined_healthy.xlsx')
unhealthy_data = pd.read_excel('/home/labadmin/R7A_group11/augmented dataset/combined_unhealthy.xlsx')
print("Data loaded successfully.")

# Combine the data
print("Combining the data...")
data = pd.concat([healthy_data, unhealthy_data], ignore_index=True)
print("Data combined successfully.")

# Load corresponding images
print("Loading images...")
healthy_image_dir = '/home/labadmin/R7A_group11/augmented dataset/data/healthy_augmented_actual'
unhealthy_image_dir = '/home/labadmin/R7A_group11/augmented dataset/data/unhealthy_augmented_actual'

# Load the pre-trained VGG16 model
print("Loading pre-trained VGG16 model...")
vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
print("VGG16 model loaded successfully.")

# Function to extract features from images using the VGG16 model
def extract_features_from_images(image_paths):
    features = []
    for image_path in image_paths:
        print("Processing image:", image_path)  # Debugging statement
        img = image.load_img(image_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        feature = vgg_model.predict(img_array).flatten()
        features.append(feature)
    return features

# Load features and labels
print("Loading features and labels...")
healthy_image_paths = [os.path.join(healthy_image_dir, img) for img in os.listdir(healthy_image_dir)]
unhealthy_image_paths = [os.path.join(unhealthy_image_dir, img) for img in os.listdir(unhealthy_image_dir)]
features = extract_features_from_images(healthy_image_paths + unhealthy_image_paths)
labels = np.hstack([np.zeros(len(healthy_image_paths)), np.ones(len(unhealthy_image_paths))])
print("Features and labels loaded successfully.")

# Define the classifier
print("Defining the Random Forest classifier...")
classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Perform cross-validation
print("Performing cross-validation...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(classifier, features, labels, cv=cv)

# Print the cross-validation scores
print("Cross-Validation Scores:", scores)

# Calculate and print the mean and standard deviation of the scores
print("Mean Accuracy:", np.mean(scores))
print("Standard Deviation of Accuracy:", np.std(scores))
