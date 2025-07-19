import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
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

# Initialize individual models
svm_model = SVC(kernel='linear')
decision_tree_model = DecisionTreeClassifier()
softmax_model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)

# Fit individual models
svm_model.fit(X_train_scaled, y_train)
decision_tree_model.fit(X_train_scaled, y_train)
softmax_model.fit(X_train_scaled, y_train)

# Create Voting Classifier with the individual models
voting_clf = VotingClassifier(estimators=[('svm', svm_model), ('decision_tree', decision_tree_model), ('softmax', softmax_model)], voting='hard')

# Train the Voting Classifier
voting_clf.fit(X_train_scaled, y_train)

# Make predictions on the test set
svm_predictions = svm_model.predict(X_test_scaled)
decision_tree_predictions = decision_tree_model.predict(X_test_scaled)
softmax_predictions = softmax_model.predict(X_test_scaled)
voting_predictions = voting_clf.predict(X_test_scaled)

# Print out individual predictions
print("SVM Predictions:", svm_predictions)
print("Decision Tree Predictions:", decision_tree_predictions)
print("Softmax Predictions:", softmax_predictions)
print("Voting Predictions:", voting_predictions)

# Calculate accuracy
accuracy = accuracy_score(y_test, voting_predictions)
print("Ensemble Learning Accuracy:", accuracy)

# Save the trained ensemble model
ensemble_model_save_path = "ensemble_model.pkl"
joblib.dump(voting_clf, ensemble_model_save_path)
print("Ensemble model saved at:", ensemble_model_save_path)
