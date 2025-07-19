import os
import numpy as np
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing import image
import joblib  # Import joblib directly to load the trained model

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
def predict_image_healthiness(image_path, classifier):
    # Extract features from the input image
    image_features = extract_features_from_image(image_path)
    # Make prediction using the trained classifier
    prediction = classifier.predict([image_features])[0]
    filename = os.path.basename(image_path)
    if prediction == 0:
        return filename, "Healthy"
    else:
        return filename, "Unhealthy"

# Load the trained Random Forest classifier
classifier = joblib.load('trained_model.pkl')

# Test the function with the specified input image
test_image_path = "/home/labadmin/R7A_group11/augmented dataset/healthy_augmented_actual/DSC_4891_blurred.jpg"
filename, result = predict_image_healthiness(test_image_path, classifier)
print("Prediction for", filename, ":", result)
