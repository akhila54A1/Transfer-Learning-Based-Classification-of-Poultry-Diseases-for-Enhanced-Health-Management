import os
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing import image
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D

# Step 1: Load data from Excel sheets
healthy_data = pd.read_excel('/home/labadmin/R7A_group11/augmented dataset/combined_healthy.xlsx')
unhealthy_data = pd.read_excel('/home/labadmin/R7A_group11/augmented dataset/combined_unhealthy.xlsx')
img_width = 224  # Specify your desired width
img_height = 224 

# Step 2: Load corresponding images
healthy_image_dir = '/home/labadmin/R7A_group11/augmented dataset/data/healthy_augmented_actual'
unhealthy_image_dir = '/home/labadmin/R7A_group11/augmented dataset/data/unhealthy_augmented_actual'

def load_images_from_directory(image_dir, label):
    image_data = []
    labels = []
    for img_file in os.listdir(image_dir):
        img_path = os.path.join(image_dir, img_file)
        img = image.load_img(img_path, target_size=(img_width, img_height))
        img_array = image.img_to_array(img)
        image_data.append(img_array)
        labels.append(label)
    return image_data, labels

healthy_image_data, healthy_labels = load_images_from_directory(healthy_image_dir, label=0)  # Healthy
unhealthy_image_data, unhealthy_labels = load_images_from_directory(unhealthy_image_dir, label=1)  # Unhealthy

# Combine healthy and unhealthy image data and labels
image_data = np.concatenate([healthy_image_data, unhealthy_image_data], axis=0)
labels = np.concatenate([healthy_labels, unhealthy_labels], axis=0)

# Step 3: Preprocess data
# You may need to preprocess the Excel data as required (e.g., normalization, handling missing values)

# Step 4: Define and train a model
# Assuming a simple CNN model
model = Sequential([
    Conv2D(16, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)),
    MaxPooling2D(2, 2),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(image_data, labels, test_size=0.2, random_state=42)

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Step 5: Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print("Test Accuracy:", accuracy)

# Step 6: Use the trained model for prediction
# Assuming `test_image` is the image you want to predict on
test_image = image.load_img("/home/labadmin/R7A_group11/augmented dataset/data/unhealthy_augmented_actual/_MG_5458_shifted.jpg", target_size=(img_width, img_height))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)  # Add batch dimension

prediction = model.predict(test_image)
if prediction < 0.5:
    print("Healthy")
else:
    print("Unhealthy")
