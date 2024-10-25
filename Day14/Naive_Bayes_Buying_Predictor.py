# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
# extracts the feature variables from the DataFrame. It uses the .iloc method to select all rows (denoted by :) and columns with indices 2 and 3 (Python uses 0-based indexing, so these are the third and fourth columns: age and salary). .values converts the selected data into a NumPy array, which is a common format for feature variables in machine learning.
X = dataset[['Age', 'EstimatedSalary']].values

# extracts the target variable (Purchased) from the DataFrame. It selects all rows and the last column (denoted by -1) of the DataFrame and converts it into a NumPy array. 
# Just a different syntax
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set (80%) and Test set (20%)
# random_state=0 sets a random seed to ensure reproducibility. Setting this parameter to a specific value (e.g., 0) ensures that every time you run the code, the same split will be generated. This is helpful for reproducibility in machine learning experiments.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Feature Scaling
# To ensure that all the features are on a similar scale (e.g. age has scale of say 1-100, but salary will be in thousands
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

# The fit_transform method computes the mean and standard deviation of each feature in the training set and then scales the features based on these statistics.
#After scaling, the feature variables in X_train will have a mean of 0 and a standard deviation of 1.
X_train = sc.fit_transform(X_train)
# print(X_train)

# This line scales the feature variables in the testing set (X_test). However, it doesn't recompute the mean and standard deviation; instead, it uses the mean and standard deviation that were computed from the training set during the fit_transform step. This ensures that the same scaling transformation is applied consistently to both the training and testing sets.
# Why not fit_transform() here?
# Test data is unseen data, so if we apply fit() here, it will also become 'seen'.
# That is, the model will train itself on the testing data also, which is not what we want.
# We want test data to remain 'useen' by our model
# May result into overfitting: Model works well on test data, but poorly on other unseen data.
X_test = sc.transform(X_test)
# print(X_test)

# Training the Naive Bayes model on the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
print(y_pred) # Our predicted values
print(y_test) # Actual values

# After executing these lines, the classifier object will have learned the relationships between the features and the target variable based on the Gaussian Naive Bayes model. You can then use this trained classifier to make predictions on new data or evaluate its performance on the test data.

# How did we perform? 


# Making the Confusion Matrix
# In this code snippet, you are using scikit-learn's confusion_matrix and accuracy_score functions to evaluate the performance of your Gaussian Naive Bayes classifier on the test data. 

from sklearn.metrics import confusion_matrix, accuracy_score
# ac = accuracy_score(y_test, y_pred): This line calculates the accuracy score of your classifier's predictions. Here's how it works:

# y_test contains the actual target values (ground truth) for the test set.
# y_pred contains the predicted target values made by your Gaussian Naive Bayes classifier on the test set.
# The accuracy_score function compares y_test and y_pred, and it calculates the ratio of correctly predicted instances to the total number of instances in the test set. This ratio is the accuracy of your classifier on the test data.

ac = accuracy_score(y_test,y_pred)

# cm = confusion_matrix(y_test, y_pred): This line calculates the confusion matrix for your classifier's predictions. Here's what it does:

# y_test contains the actual target values (ground truth) for the test set.
# y_pred contains the predicted target values made by your Gaussian Naive Bayes classifier # on the test set.
# The confusion_matrix function takes y_test and y_pred as inputs and computes a matrix that shows the true positive, true negative, false positive, and false negative values.
cm = confusion_matrix(y_test, y_pred)

# ac: This variable contains the accuracy score, which quantifies the overall correctness of your classifier's predictions on the test data.
# cm: This variable contains the confusion matrix, which provides a detailed breakdown of how your classifier's predictions compare to the actual ground truth values, allowing you to assess its performance in more detail.

print(ac)
print(cm)