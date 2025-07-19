import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix
from tensorflow.keras.preprocessing import image
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D

# Step 1: Load data from Excel sheets
print("Loading data from Excel sheets...")
healthy_data = pd.read_excel('/home/labadmin/R7A_group11/augmented dataset/combined_healthy.xlsx')
unhealthy_data = pd.read_excel('/home/labadmin/R7A_group11/augmented dataset/combined_unhealthy.xlsx')
img_width = 224  # Specify your desired width
img_height = 224 

# Step 2: Load corresponding images
print("Loading corresponding images...")
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
print("Defining and training the model...")
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
print("Splitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(image_data, labels, test_size=0.2, random_state=42)

# Train the model
print("Training the model...")
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Step 5: Evaluate the model
print("Evaluating the model...")
loss, accuracy = model.evaluate(X_test, y_test)
print("Test Accuracy:", accuracy)

# Predict probabilities for ROC curve
print("Predicting probabilities for ROC curve...")
y_probs = model.predict(X_test)

# Compute ROC curve and AUC
print("Computing ROC curve and AUC...")
fpr, tpr, thresholds = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)
print("ROC AUC:", roc_auc)

# Plot ROC curve
print("Plotting ROC curve...")
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# Compute confusion matrix
print("Computing confusion matrix...")
y_pred = (y_probs > 0.5).astype(int)
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Plot confusion matrix
print("Plotting confusion matrix...")
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ['Healthy', 'Unhealthy'])
plt.yticks(tick_marks, ['Healthy', 'Unhealthy'])
plt.ylabel('True label')
plt.xlabel('Predicted label')
for i in range(2):
    for j in range(2):
        plt.text(j, i, format(conf_matrix[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if conf_matrix[i, j] > conf_matrix.max() / 2. else "black")
plt.show()

# Step 6: Use the trained model for prediction
# Assuming `test_image` is the image you want to predict on
print("Making predictions on a sample image...")
test_image = image.load_img("/home/labadmin/R7A_group11/augmented dataset/data/unhealthy_augmented_actual/_MG_5458_shifted.jpg", target_size=(img_width, img_height))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)  # Add batch dimension

prediction = model.predict(test_image)
if prediction < 0.5:
    print("Prediction: Healthy")
else:
    print("Prediction: Unhealthy")
