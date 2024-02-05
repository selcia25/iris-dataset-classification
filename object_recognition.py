# Import necessary modules - pip install scikit-learn pandas

# Load dataset using pandas

import pandas as pd

iris_df = pd.read_csv('Iris.csv')
print("----------------------------------------------------------------------------------")
print("Iris Dataset")
print("----------------------------------------------------------------------------------")
print(iris_df)
print("----------------------------------------------------------------------------------")

# Explore the data using head(), describe() and info() options

print("First Few Rows in Iris Dataset")
print("----------------------------------------------------------------------------------")
print(iris_df.head())  # first few rows
print("----------------------------------------------------------------------------------")
print("Information about Iris Dataset")
print("----------------------------------------------------------------------------------")
print(iris_df.info())  # check missing values
print("----------------------------------------------------------------------------------")
print("Summary Statistics about Iris Dataset")
print("----------------------------------------------------------------------------------")
print(iris_df.describe())  # summary statistics
print("----------------------------------------------------------------------------------")

# Data preprocessing

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# splitting features and target variables
X = iris_df.drop(['Id', 'Species'], axis=1)
y = iris_df['Species']

# encode the target variables to numerical variables
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# choose classification algorithm
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# train the classifier
clf.fit(X_train, y_train)

# evaluate model's performance
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# predictions on testing set
y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy score: {accuracy:.2f}')
print("----------------------------------------------------------------------------------")
report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
print('Classification Report:\n', report)
print("----------------------------------------------------------------------------------")
conf_matrix = confusion_matrix(y_test, y_pred)
print('Confusion matrix:\n', conf_matrix)
print("----------------------------------------------------------------------------------")
# Predictions

new_data = [[5.1, 3.5, 1.4, 0.2]]
print(f"Input Data: {new_data}")
predicted_class = label_encoder.inverse_transform(clf.predict(new_data))
print(predicted_class)
print(f'Predicted class for new data: {predicted_class[0]}')
print("----------------------------------------------------------------------------------")
new_data = [[8.1, 3.5, 5.4, 0.2]]
print(f"Input Data: {new_data}")
predicted_class = label_encoder.inverse_transform(clf.predict(new_data))
print(f'Predicted class for new data: {predicted_class[0]}')
print("----------------------------------------------------------------------------------")