import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RANSACRegressor

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/'
                 'housing/housing.data',
                 header=None,
                 sep='\s+')

df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS',
              'NOX', 'RM', 'AGE', 'DIS', 'RAD',
              'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
print('Dataset excerpt:\n\n', df.head())

X, y = df.iloc[:, 1:].values, df.iloc[:, 0].values

X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.3, random_state=0)

X = df[['RM']].values
y = df['MEDV'].values

ransac = RANSACRegressor(LinearRegression(),
                         max_trials=1000,
                         min_samples=50,
                         residual_metric=lambda x: np.sum(np.abs(x), axis=1),
                         residual_threshold=5.0,
                         random_state=0)
ransac.fit(X, y)
inlier_mask = ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)

line_X = np.arange(3, 10, 1)
line_y_ransac = ransac.predict(line_X[:, np.newaxis])
plt.scatter(X[inlier_mask], y[inlier_mask],
            c='blue', marker='o', label='Inliers')
plt.scatter(X[outlier_mask], y[outlier_mask],
            c='lightgreen', marker='s', label='Outliers')
plt.plot(line_X, line_y_ransac, color='red')
plt.xlabel('Average number of rooms [RM]')
plt.ylabel('Price in $1000\'s [MEDV]')
plt.legend(loc='upper left')
# plt.show()
plt.close()

X = df.iloc[:, :-1].values
y = df['MEDV'].values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0)

slr = LinearRegression()

slr.fit(X_train, y_train)
y_train_pred = slr.predict(X_train)
# y_test_pred = slr.predict(X_test)
#
## Sommes des moindres carrés
# from sklearn.metrics import mean_squared_error
# print('MSE train: %.3f, test: %.3f' % (
#        mean_squared_error(y_train, y_train_pred),
#        mean_squared_error(y_test, y_test_pred)))
# print('R^2 train: %.3f, test: %.3f' % (
#        r2_score(y_train, y_train_pred),
# r2_score(y_test, y_test_pred)))
#
#
#
## On cherche a modéliser pl
# X = df[['LSTAT']].values
# y = df['MEDV'].values
#
# regr = LinearRegression()
#
## create quadratic features
# quadratic = PolynomialFeatures(degree=2)
# cubic = PolynomialFeatures(degree=3)
# X_quad = quadratic.fit_transform(X)
# X_cubic = cubic.fit_transform(X)
#
## fit features
# X_fit = np.arange(X.min(), X.max(), 1)[:, np.newaxis]
#
# regr = regr.fit(X, y)
# y_lin_fit = regr.predict(X_fit)
# linear_r2 = r2_score(y, regr.predict(X))
#
# regr = regr.fit(X_quad, y)
# y_quad_fit = regr.predict(quadratic.fit_transform(X_fit))
# quadratic_r2 = r2_score(y, regr.predict(X_quad))
#
# regr = regr.fit(X_cubic, y)
# y_cubic_fit = regr.predict(cubic.fit_transform(X_fit))
# cubic_r2 = r2_score(y, regr.predict(X_cubic))
#
#
## plot results
# plt.scatter(X, y, label='training points', color='lightgray')
#
# plt.plot(X_fit, y_lin_fit,
#         label='linear (d=1), $R^2=%.2f$' % linear_r2,
#         color='blue',
#         lw=2,
#         linestyle=':')
#
# plt.plot(X_fit, y_quad_fit,
#         label='quadratic (d=2), $R^2=%.2f$' % quadratic_r2,
#         color='red',
#         lw=2,
#         linestyle='-')
#
# plt.plot(X_fit, y_cubic_fit,
#         label='cubic (d=3), $R^2=%.2f$' % cubic_r2,
#         color='green',
#         lw=2,
#         linestyle='--')
#
# plt.xlabel('% lower status of the population [LSTAT]')
# plt.ylabel('Price in $1000\'s [MEDV]')
# plt.legend(loc='upper right')
#
# plt.show()
#
#
#
# print('Transforming the dataset')
# X = df[['LSTAT']].values
# y = df['MEDV'].values
#
## transform features
# X_log = np.log(X)
# y_sqrt = np.sqrt(y)
#
## fit features
# X_fit = np.arange(X_log.min()-1, X_log.max()+1, 1)[:, np.newaxis]
#
# regr = regr.fit(X_log, y_sqrt)
# y_lin_fit = regr.predict(X_fit)
# linear_r2 = r2_score(y_sqrt, regr.predict(X_log))
#
## plot results
# plt.scatter(X_log, y_sqrt, label='training points', color='lightgray')
#
# plt.plot(X_fit, y_lin_fit,
#         label='linear (d=1), $R^2=%.2f$' % linear_r2,
#         color='blue',
#         lw=2)
#
# plt.xlabel('log(% lower status of the population [LSTAT])')
# plt.ylabel('$\sqrt{Price \; in \; \$1000\'s [MEDV]}$')
# plt.legend(loc='lower left')
#
# plt.show()
