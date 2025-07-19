import os
import numpy as np
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing import image
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
import joblib  # Import joblib directly to save the trained model

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

# Load features and labels
healthy_image_dir = '/home/labadmin/R7A_group11/augmented dataset/data/healthy_augmented_actual'
unhealthy_image_dir = '/home/labadmin/R7A_group11/augmented dataset/data/unhealthy_augmented_actual'

healthy_image_paths = [os.path.join(healthy_image_dir, img) for img in os.listdir(healthy_image_dir)]
unhealthy_image_paths = [os.path.join(unhealthy_image_dir, img) for img in os.listdir(unhealthy_image_dir)]

# Split the dataset into train and test sets
healthy_train, healthy_test = train_test_split(healthy_image_paths, test_size=0.3, random_state=42)
unhealthy_train, unhealthy_test = train_test_split(unhealthy_image_paths, test_size=0.3, random_state=42)

train_paths = healthy_train + unhealthy_train
test_paths = healthy_test + unhealthy_test

# Extract features for training
train_features = [extract_features_from_image(img_path) for img_path in train_paths]
test_features = [extract_features_from_image(img_path) for img_path in test_paths]

X_train = np.vstack(train_features)
y_train = np.hstack([np.zeros(len(healthy_train)), np.ones(len(unhealthy_train))])
X_test = np.vstack(test_features)
y_test = np.hstack([np.zeros(len(healthy_test)), np.ones(len(unhealthy_test))])

# Hyperparameter tuning
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20, 30]
}

rf_classifier = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Get the best parameters and print the accuracy
best_params = grid_search.best_params_
print("Best Parameters:", best_params)

# Evaluate accuracy on the test set
best_rf_classifier = grid_search.best_estimator_
y_pred = best_rf_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy after optimization:", accuracy)

# Save the trained model as a pickle file
joblib.dump(best_rf_classifier, 'trained_model.pkl')

print("\nTrained and optimized model saved successfully as 'optimized_model.pkl'.")

# Test the function with the specified input image
test_image_path = "/home/labadmin/R7A_group11/augmented dataset/data/unhealthy_augmented_actual/_MG_5543_blurred.jpg"
filename, result = predict_image_healthiness(test_image_path, best_rf_classifier)
print("Prediction for", filename, ":", result)
