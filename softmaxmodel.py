import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# Load features and labels from Excel sheets
train_healthy_path = "/home/labadmin/R7A_group11/augmented dataset/train_healthy_feature.xlsx"
train_unhealthy_path = "/home/labadmin/R7A_group11/augmented dataset/train_unhealthy_feature.xlsx"
test_healthy_path = "/home/labadmin/R7A_group11/augmented dataset/test_healthy_feature.xlsx"
test_unhealthy_path = "/home/labadmin/R7A_group11/augmented dataset/test_unhealthy_feature.xlsx"

# Read training data
train_healthy_df = pd.read_excel(train_healthy_path)
train_unhealthy_df = pd.read_excel(train_unhealthy_path)

# Read testing data
test_healthy_df = pd.read_excel(test_healthy_path)
test_unhealthy_df = pd.read_excel(test_unhealthy_path)

# Add labels to the dataframes (1 for healthy, 0 for unhealthy)
train_healthy_df['Label'] = 1
train_unhealthy_df['Label'] = 0
test_healthy_df['Label'] = 1
test_unhealthy_df['Label'] = 0

# Concatenate the dataframes
train_combined_df = pd.concat([train_healthy_df, train_unhealthy_df], ignore_index=True)
test_combined_df = pd.concat([test_healthy_df, test_unhealthy_df], ignore_index=True)

# Drop non-numerical columns (if any)
X_train = train_combined_df.drop(['Label'], axis=1)
y_train = train_combined_df['Label']
X_test = test_combined_df.drop(['Label'], axis=1)
y_test = test_combined_df['Label']

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train the Softmax Regression model
softmax_model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
softmax_model.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = softmax_model.predict(X_test_scaled)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Softmax Regression Accuracy:", accuracy)

# Save the trained model
model_save_path = "softmax_regression_model.pkl"
joblib.dump(softmax_model, model_save_path)
print("Model saved at:", model_save_path)
