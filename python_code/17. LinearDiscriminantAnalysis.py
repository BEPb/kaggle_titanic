"""
Python 3.10 Linear Discriminant Analysis program with pre-processing of kaggle titanic competition data
File name: LinearDiscriminantAnalysis.py

Version: 0.1
Author: Andrej Marinchenko
Date: 2023-01-11
"""

import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

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

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=33)

############################################## Train the model #########################################################
model = LinearDiscriminantAnalysis()
model.fit(X_train, y_train)

# Evaluate the logistic regression classifier
scores = cross_val_score(model, X_val, y_val, cv=5)
print("Accuracy of linear regression classifier: ", scores.mean())

# Make predictions on the test set
y_pred = model.predict(X_test)

# Save the predictions to a CSV file
output = pd.DataFrame({'PassengerId': test_df['PassengerId'], 'Survived': y_pred})
output['Survived'] = output['Survived'].astype(int)
output.to_csv('17.submission-lda-0.76555.csv', index=False)

print(output)
print('Correlation with ideal submission:', output['Survived'].corr(result_df['Survived']))
result_df['percent'] = result_df['Survived'] == output['Survived']
print('percent: \n', (result_df['percent'].value_counts('True')))
print('Real score on submission: 0.76555')

# ################################################### Plot ##############################################################
# import matplotlib.pyplot as plt
# import matplotlib as mpl
# from matplotlib import colors
#
# cmap = colors.LinearSegmentedColormap(
#     "red_blue_classes",
#     {
#         "red": [(0, 1, 1), (1, 0.7, 0.7)],
#         "green": [(0, 0.7, 0.7), (1, 0.7, 0.7)],
#         "blue": [(0, 0.7, 0.7), (1, 1, 1)],
#     },
# )
# plt.cm.register_cmap(cmap=cmap)
#
# ################################################# Plot functions #####################################################
#
# from scipy import linalg
#
#
# def plot_data(lda, X, y, y_pred, fig_index):
#     splot = plt.subplot(2, 2, fig_index)
#     if fig_index == 1:
#         plt.title("Linear Discriminant Analysis")
#         plt.ylabel("Data with\n fixed covariance")
#     elif fig_index == 2:
#         plt.title("Quadratic Discriminant Analysis")
#     elif fig_index == 3:
#         plt.ylabel("Data with\n varying covariances")
#
#     tp = y == y_pred  # True Positive
#     tp0, tp1 = tp[y == 0], tp[y == 1]
#     X0, X1 = X[y == 0], X[y == 1]
#     X0_tp, X0_fp = X0[tp0], X0[~tp0]
#     X1_tp, X1_fp = X1[tp1], X1[~tp1]
#
#     # class 0: dots
#     plt.scatter(X0_tp[:, 0], X0_tp[:, 1], marker=".", color="red")
#     plt.scatter(X0_fp[:, 0], X0_fp[:, 1], marker="x", s=20, color="#990000")  # dark red
#
#     # class 1: dots
#     plt.scatter(X1_tp[:, 0], X1_tp[:, 1], marker=".", color="blue")
#     plt.scatter(
#         X1_fp[:, 0], X1_fp[:, 1], marker="x", s=20, color="#000099"
#     )  # dark blue
#
#     # class 0 and 1 : areas
#     nx, ny = 200, 100
#     x_min, x_max = plt.xlim()
#     y_min, y_max = plt.ylim()
#     xx, yy = np.meshgrid(np.linspace(x_min, x_max, nx), np.linspace(y_min, y_max, ny))
#     Z = lda.predict_proba(np.c_[xx.ravel(), yy.ravel()])
#     Z = Z[:, 1].reshape(xx.shape)
#     plt.pcolormesh(
#         xx, yy, Z, cmap="red_blue_classes", norm=colors.Normalize(0.0, 1.0), zorder=0
#     )
#     plt.contour(xx, yy, Z, [0.5], linewidths=2.0, colors="white")
#
#     # means
#     plt.plot(
#         lda.means_[0][0],
#         lda.means_[0][1],
#         "*",
#         color="yellow",
#         markersize=15,
#         markeredgecolor="grey",
#     )
#     plt.plot(
#         lda.means_[1][0],
#         lda.means_[1][1],
#         "*",
#         color="yellow",
#         markersize=15,
#         markeredgecolor="grey",
#     )
#
#     return splot
#
#
# def plot_ellipse(splot, mean, cov, color):
#     v, w = linalg.eigh(cov)
#     u = w[0] / linalg.norm(w[0])
#     angle = np.arctan(u[1] / u[0])
#     angle = 180 * angle / np.pi  # convert to degrees
#     # filled Gaussian at 2 standard deviation
#     ell = mpl.patches.Ellipse(
#         mean,
#         2 * v[0] ** 0.5,
#         2 * v[1] ** 0.5,
#         angle=180 + angle,
#         facecolor=color,
#         edgecolor="black",
#         linewidth=2,
#     )
#     ell.set_clip_box(splot.bbox)
#     ell.set_alpha(0.2)
#     splot.add_artist(ell)
#     splot.set_xticks(())
#     splot.set_yticks(())
#
#
# def plot_lda_cov(lda, splot):
#     plot_ellipse(splot, lda.means_[0], lda.covariance_, "red")
#     plot_ellipse(splot, lda.means_[1], lda.covariance_, "blue")
#
#
# def plot_qda_cov(qda, splot):
#     plot_ellipse(splot, qda.means_[0], qda.covariance_[0], "red")
#     plot_ellipse(splot, qda.means_[1], qda.covariance_[1], "blue")
#
# #####################################################################################################################
#
# plt.figure(figsize=(10, 8), facecolor="white")
# plt.suptitle(
#     "Linear Discriminant Analysis vs Quadratic Discriminant Analysis",
#     y=0.98,
#     fontsize=15,
# )
#
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
#
# for i, (X, y) in enumerate(X_train, y_train):
#     # Linear Discriminant Analysis
#     lda = LinearDiscriminantAnalysis(solver="svd", store_covariance=True)
#     y_pred = lda.fit(X, y).predict(X)
#     splot = plot_data(lda, X, y, y_pred, fig_index=2 * i + 1)
#     plot_lda_cov(lda, splot)
#     plt.axis("tight")
#
#     # Quadratic Discriminant Analysis
#     qda = QuadraticDiscriminantAnalysis(store_covariance=True)
#     y_pred = qda.fit(X, y).predict(X)
#     splot = plot_data(qda, X, y, y_pred, fig_index=2 * i + 2)
#     plot_qda_cov(qda, splot)
#     plt.axis("tight")
#
# plt.tight_layout()
# plt.subplots_adjust(top=0.92)
# plt.show()