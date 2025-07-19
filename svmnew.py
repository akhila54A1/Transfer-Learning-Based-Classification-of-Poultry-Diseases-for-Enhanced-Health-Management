import os
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Function to load data and label it
def load_and_label_data(file_paths, label):
    dfs = []
    for file_path in file_paths:
        df = pd.read_excel(file_path)
        df['Label'] = label  # Assign label to the 'Label' column
        # Drop the 'filename' column
        df = df.drop('filename', axis=1)
        # Map 'convex' to 1 and 'concave' to 0
        df['Concavity'] = df['Concavity'].map({'convex': 1, 'concave': 0})
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

# Directory paths
excel_sheets_dir = "/home/labadmin/R7A_group11/segmented dataset/excel_sheets/"

# Construct full file paths
train_paths = {
    "stunned": [os.path.join(excel_sheets_dir, "Stunned_excels/train", file) for file in os.listdir(os.path.join(excel_sheets_dir, "Stunned_excels/train"))],
    "slipped_tendons": [os.path.join(excel_sheets_dir, "Slipped_excels/train", file) for file in os.listdir(os.path.join(excel_sheets_dir, "Slipped_excels/train"))],
    "drooping_neck": [os.path.join(excel_sheets_dir, "Drooping_excels/train", file) for file in os.listdir(os.path.join(excel_sheets_dir, "Drooping_excels/train"))]
}

test_paths = {
    "stunned": [os.path.join(excel_sheets_dir, "Stunned_excels/test", file) for file in os.listdir(os.path.join(excel_sheets_dir, "Stunned_excels/test"))],
    "slipped_tendons": [os.path.join(excel_sheets_dir, "Slipped_excels/test", file) for file in os.listdir(os.path.join(excel_sheets_dir, "Slipped_excels/test"))],
    "drooping_neck": [os.path.join(excel_sheets_dir, "Drooping_excels/test", file) for file in os.listdir(os.path.join(excel_sheets_dir, "Drooping_excels/test"))]
}

# Load and label training data
train_combined_df = {}
for disease, paths in train_paths.items():
    train_combined_df[disease] = load_and_label_data(paths, disease)

# Load and label testing data
test_combined_df = {}
for disease, paths in test_paths.items():
    test_combined_df[disease] = load_and_label_data(paths, disease)

# Concatenate training and testing dataframes
train_df = pd.concat(train_combined_df.values(), ignore_index=True)
test_df = pd.concat(test_combined_df.values(), ignore_index=True)

# Separate features and labels
X_train = train_df.drop(['Label'], axis=1)
y_train = train_df['Label']
X_test = test_df.drop(['Label'], axis=1)
y_test = test_df['Label']

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the parameter grid
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
}

# Create the SVM classifier
svm = SVC()

# Reduce the number of cross-validation folds (e.g., 3-fold cross-validation)
cv = 3

# Perform grid search with reduced parameters and folds
grid_search = GridSearchCV(estimator=svm, param_grid=param_grid, cv=cv, n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

# Print best hyperparameters and corresponding score
print("Best parameters found:", grid_search.best_params_)
print("Best cross-validation score:", grid_search.best_score_)

# Get the test accuracy using the best model
test_accuracy = grid_search.score(X_test_scaled, y_test)
print("Test set accuracy:", test_accuracy)
