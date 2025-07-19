import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib

# Function to load data and label it
def load_and_label_data(file_paths, label):
    dfs = []
    for file_path in file_paths:
        print(f"Loading data from file: {file_path}")
        df = pd.read_excel(file_path)
        print(f"Loaded {len(df)} samples")
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
    print(f"Loading training data for {disease}")
    train_combined_df[disease] = load_and_label_data(paths, disease)

# Load and label testing data
test_combined_df = {}
for disease, paths in test_paths.items():
    print(f"Loading testing data for {disease}")
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

# Initialize and train the Decision Tree model
decision_tree_model = DecisionTreeClassifier()
decision_tree_model.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = decision_tree_model.predict(X_test_scaled)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Decision Tree Accuracy:", accuracy)

# Save the trained model
# model_save_path = "decision_tree_model.pkl"
# joblib.dump(decision_tree_model, model_save_path)
# print("Model saved at:", model_save_path)
