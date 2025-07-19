import os
import numpy as np
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing import image
from sklearn.ensemble import RandomForestClassifier
import joblib  # Import joblib directly to save the trained model
from sklearn.model_selection import train_test_split

# Load the pre-trained VGG16 model
vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Function to extract features from an image using the VGG16 model
def extract_features_from_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = vgg_model.predict(img_array)
    features_flattened = features.flatten()
    return features_flattened

# Function to predict whether an image is healthy or unhealthy, and return the filename along with the prediction
def predict_image_healthiness(image_path):
    # Extract features from the input image
    image_features = extract_features_from_image(image_path)
    # Make prediction using the trained classifier
    prediction = classifier.predict([image_features])[0]
    filename = os.path.basename(image_path)
    if prediction == 0:
        return filename, "Healthy"
    else:
        return filename, "Unhealthy"

# Load features and labels
image_dir = '/home/labadmin/R7A_group11/augmented dataset/data'

image_paths = []
labels = []

for label, class_dir in enumerate(['new_healthy_augmented_actual', 'unhealthy_augmented_actual']):
    class_path = os.path.join(image_dir, class_dir)
    class_images = [os.path.join(class_path, img) for img in os.listdir(class_path)]
    image_paths.extend(class_images)
    labels.extend([label] * len(class_images))

# Split dataset into training and testing sets
train_paths, test_paths, train_labels, test_labels = train_test_split(image_paths, labels, test_size=0.2, random_state=42)

# Print filenames while training
print("Training with images:")
for img_path in train_paths:
    print("Processing:", img_path)
    _ = extract_features_from_image(img_path)

# Extract features for training
train_features = [extract_features_from_image(img_path) for img_path in train_paths]

# Define and train the Random Forest classifier
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(train_features, train_labels)

print("\nTraining completed.")

# Test the function with the specified input image
test_image_path = test_paths[0]
filename, result = predict_image_healthiness(test_image_path)
print("Prediction for", filename, ":", result)

# Print filenames while testing
print("\nTesting with images:")
for img_path in test_paths:
    print("Processing:", img_path)

# Extract features for testing
test_features = [extract_features_from_image(img_path) for img_path in test_paths]

# Predict labels for the test set
predicted_labels = classifier.predict(test_features)

# Calculate accuracy
accuracy = np.mean(predicted_labels == test_labels) * 100

print("\nAccuracy:", accuracy, "%")
