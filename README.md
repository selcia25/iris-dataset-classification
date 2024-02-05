# Iris Dataset Classification using Random Forest

This repository contains a Python script for classifying the Iris dataset using the Random Forest algorithm. The script covers data loading, exploration, preprocessing, model training, evaluation, and making predictions for new data points.

## Usage
1. **Clone the repository:**
   ```bash
   git clone https://github.com/selcia25/iris-dataset-classification.git
   cd iris-dataset-classification
   ```

2. **Install Dependencies:**
   ```bash
   pip install scikit-learn pandas
   ```

3. **Download Dataset:**
   - Download the Iris dataset (Iris.csv) or replace it with your dataset.
   - Update the file name in the script accordingly.

4. **Run the Script:**
   ```bash
   python iris_classification.py
   ```

## Dependencies
- scikit-learn
- pandas

## Script Overview
1. **Load and Explore Dataset:**
   - Load the Iris dataset using pandas.

2. **Explore Data:**
   - Display first few rows, information, and summary statistics of the dataset.

3. **Data Preprocessing:**
   - Split features and target variables.
   - Encode target variables to numerical values.
   - Split the data into training and testing sets.

4. **Choose Classification Algorithm and Train Model:**
   - Use the Random Forest classifier with 100 estimators.

5. **Evaluate Model's Performance:**
   - Display accuracy score, classification report, and confusion matrix.

6. **Make Predictions for New Data Points:**
   - Provide sample data points and display predicted classes.

This script serves as a basic template for classification tasks on the Iris dataset and can be extended for other datasets or machine learning tasks.
