import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.optimizers import Adam
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform
from scikeras.wrappers import KerasClassifier

# Load the pre-trained VGG16 model
vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the VGG16 model layers
for layer in vgg_model.layers:
    layer.trainable = False

# Define a function to create the model
def create_model(learning_rate=0.0001, dropout_rate=0.5):
    model = Sequential()
    model.add(vgg_model)
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(dropout_rate))  # Adjust dropout rate directly in the model
    model.add(Dense(1, activation='sigmoid'))

    # Define custom optimizer with desired learning rate
    optimizer = Adam(lr=learning_rate)  # Set the learning rate here
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Create KerasClassifier wrapper for use in RandomizedSearchCV
model = KerasClassifier(build_fn=create_model, verbose=0)

# Define directories for training and testing data
train_dir = '/home/labadmin/R7A_group11/augmented dataset/data'
test_dir = '/home/labadmin/R7A_group11/augmented dataset/data'

# Load and preprocess data using ImageDataGenerator
train_data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)
test_data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = train_data_generator.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary')

test_generator = test_data_generator.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary')

# Define the search space for hyperparameters
param_dist = {
    'learning_rate': uniform(loc=0.0001, scale=0.01),  # Uniform distribution for learning rate
    'dropout_rate': [0.3, 0.4, 0.5]  # Discrete values for dropout rate
}

# Perform RandomizedSearchCV
random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, n_iter=10, cv=2)
random_result = random_search.fit(train_generator, epochs=1, validation_data=test_generator, steps_per_epoch=len(train_generator), validation_steps=len(test_generator))

# Get the best parameters and print the accuracy
best_params = random_result.best_params_
print("Best Parameters:", best_params)

# Print the accuracy
best_accuracy = random_result.best_score_
print("Best Accuracy:", best_accuracy)

# Save the trained model
best_model = random_result.best_estimator_.model
best_model.save('deep_trained_model_randomized.h5')

print("\nTrained and optimized model saved successfully as 'deep_trained_model_randomized.h5'.")
