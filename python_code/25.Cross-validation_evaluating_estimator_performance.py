"""
Python 3.10 Cross-validation: evaluating estimator performance with Support Vector Machines program with pre-processing
 of kaggle titanic competition data
File name: Cross-validation_evaluating_estimator_performance.py

Version: 0.1
Author: Andrej Marinchenko
Date: 2023-01-14
"""

import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, ShuffleSplit  # random split into training and test sets and Cross-validation
from sklearn import svm  # Support Vector Machines
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

#### plot
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt


# Read in the training and test sets
train_df = pd.read_csv('../data/train.csv')
# print(len(train_df))
test_df = pd.read_csv('../data/test.csv')
# print(len(test_df))
result_df = pd.read_csv('../data/submission-titanic.csv')   # 100% result
# print(len(result_df))

###################################### Preprocess the data #############################################################
# Identify most relevant features
# You can use techniques like feature importance or correlation analysis to help you identify the most important features
relevant_features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']

# Handle missing values
imputer = SimpleImputer(strategy='most_frequent')
train_df[relevant_features] = imputer.fit_transform(train_df[relevant_features])
test_df[relevant_features] = imputer.transform(test_df[relevant_features])

# Encode categorical variables as numeric
train_df['Sex'] = train_df['Sex'].map({'male': 0, 'female': 1})
test_df['Sex'] = test_df['Sex'].map({'male': 0, 'female': 1})
train_df['Embarked'] = train_df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
test_df['Embarked'] = test_df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

# Transform skewed or non-normal features
# Instead of normalizing all of the numeric features, you could try using techniques like log transformation or
# Box-Cox transformation to make the distribution of a feature more normal
scaler = StandardScaler()
train_df[relevant_features] = scaler.fit_transform(train_df[relevant_features])
test_df[relevant_features] = scaler.transform(test_df[relevant_features])

# Split the data into features (X) and labels (y)
X_train = train_df[relevant_features]
y_train = train_df['Survived']
X_test = test_df[relevant_features]


print('X_train before split: ', X_train.shape)
print('y_train before split: ', y_train.shape)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=30)  # random split into training and test sets
print('\nX_train after split: ', X_train.shape)
print('X_valafter split:  ', X_val.shape)
print('y_train after split: ', y_train.shape)
print('y_valafter split: ', y_val.shape)
############################################## Train the model #########################################################
# model = svm.SVC(kernel='linear', C=1)
# model = svm.SVC()
model = make_pipeline(StandardScaler(), svm.SVC())
model.fit(X_train, y_train)



################################################### Plot ###############################################################
# svc_disp = RocCurveDisplay.from_estimator(model, X_val, y_val)
# plt.xlabel('X-val')
# plt.ylabel('Y-val')
# plt.title("A simple line graph")

y_score = model.decision_function(X_val)
fpr, tpr, _ = roc_curve(y_val, y_score, pos_label=model.classes_[1])
roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr)

prec, recall, _ = precision_recall_curve(y_val, y_score, pos_label=model.classes_[1])
pr_display = PrecisionRecallDisplay(precision=prec, recall=recall)
# plt.title("Roc Curve")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
roc_display.plot(ax=ax1)
pr_display.plot(ax=ax2)
plt.show()
################################################## End plot ############################################################


# Evaluate the logistic regression classifier
scores = cross_val_score(model, X_val, y_val, cv=5)  # Computing cross-validated metrics
print("\n %0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
print('\n ', cross_val_score(model, X_val, y_val, cv=5))

# Make predictions on the test set
y_pred = model.predict(X_test)

# Save the predictions to a CSV file
output = pd.DataFrame({'PassengerId': test_df['PassengerId'], 'Survived': y_pred})
output['Survived'] = output['Survived'].astype(int)
output.to_csv('25.submission-svm-0.782297.csv', index=False)

# print(output)
print('\n Correlation with ideal submission:', output['Survived'].corr(result_df['Survived']))
result_df['percent'] = result_df['Survived'] == output['Survived']
print('percent: \n', (result_df['percent'].value_counts('True')))
print('Real score on submission: 0.782297')