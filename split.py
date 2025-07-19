import pandas as pd
from sklearn.model_selection import train_test_split

# Load the feature Excel sheet
feature_excel_path = "/home/labadmin/R7A_group11/augmented dataset/combined_total_unhealthy_features.xlsx"
feature_df = pd.read_excel(feature_excel_path)

# Split the data into features (X) and labels (y)
X = feature_df.drop('filename', axis=1)  # Assuming the first column is 'filename'
y = None  # No labels

# Split the data into training and testing sets
X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

# Print the shapes of the resulting datasets
print("Training set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)

# Optionally, save the train and test datasets to new Excel files
X_train.to_excel("/home/labadmin/R7A_group11/augmented dataset/train_unhealthy_feature.xlsx", index=False)
X_test.to_excel("/home/labadmin/R7A_group11/augmented dataset/test_unhealthy_feature.xlsx", index=False)

