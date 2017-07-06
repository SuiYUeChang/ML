"""
========================================
Lasso and Elastic Net for Sparse Signals
========================================

Estimates Lasso and Elastic-Net regression models on a manually generated
sparse signal corrupted with an additive noise. Estimated coefficients are
compared with the ground-truth.

"""
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score

#总结线性回归
# http://blog.csdn.net/puqutogether/article/details/40889719
###############################################################################
# generate some sparse data to play with
np.random.seed(42)

n_samples, n_features = 50, 200
X = np.random.randn(n_samples, n_features)
coef = 3 * np.random.randn(n_features)
inds = np.arange(n_features)
np.random.shuffle(inds)
coef[inds[10:]] = 0  # sparsify coef #表示系数加入了稀疏的约束
y = np.dot(X, coef)

# add noise
y += 0.01 * np.random.normal((n_samples,))

# Split data in train set and test set
n_samples = X.shape[0]
X_train, y_train = X[:n_samples // 2], y[:n_samples // 2]
X_test, y_test = X[n_samples // 2:], y[n_samples // 2:]

###############################################################################
# Lasso
from sklearn.linear_model import Lasso

alpha = 0.1
lasso = Lasso(alpha=alpha)

y_pred_lasso = lasso.fit(X_train, y_train).predict(X_test)
r2_score_lasso = r2_score(y_test, y_pred_lasso)
print(lasso)
print("r^2 on test data : %f" % r2_score_lasso)

###############################################################################
# ElasticNet
from sklearn.linear_model import ElasticNet

enet = ElasticNet(alpha=alpha, l1_ratio=0.7)

y_pred_enet = enet.fit(X_train, y_train).predict(X_test)
r2_score_enet = r2_score(y_test, y_pred_enet)
print(enet)
print("r^2 on test data : %f" % r2_score_enet)

plt.plot(enet.coef_, color='lightgreen', linewidth=2,
         label='Elastic net coefficients')
plt.plot(lasso.coef_, color='gold', linewidth=2,
         label='Lasso coefficients')
plt.plot(coef, '--', color='navy', label='original coefficients')
plt.legend(loc='best')
plt.title("Lasso R^2: %f, Elastic Net R^2: %f"
          % (r2_score_lasso, r2_score_enet))
plt.show()

## r2
#R-Squared Calculation Example
#The calculation of R-squared requires several steps. First, assume the following set of (x, y) data points:
# (3, 40), (10, 35), (11, 30), (15, 32), (22, 19), (22, 26), (23, 24), (28, 22), (28, 18) and (35, 6).

#To calculate the R-squared, an analyst needs to have a "line of best fit" equation. 
# This equation, based on the unique date, is an equation that predicts a Y value based on a given X value. 
# In this example, assume the line of best fit is: y = 0.94x + 43.7

#With that, an analyst could compute predicted Y values. As an example, the predicted Y value for the first data point is:

#y = 0.94(3) + 43.7 = 40.88

#The entire set of predicted Y values is: 40.88, 34.3, 33.36, 29.6, 23.02, 23.02, 22.08, 17.38, 17.38 and 10.8. 
# Next, the analyst takes each data point's predicted Y value, subtracts the actual Y value and squares the result.
# For example, using the first data point:

#Error squared = (40.88 - 40) ^ 2 = 0.77

#The entire list of error's squared is: 0.77, 0.49, 11.29, 5.76, 16.16, 8.88, 3.69, 21.34, 0.38 and 23.04. 
# The sum of these errors is 91.81. Next, the analyst takes the predicted Y value and subtracts the average actual value, which is 25.2.
# Using the first data point, this is:

#(40.88 - 25.2) ^ 2 = 14.8 ^ 2 = 219.04. The analyst sums up all these differences, which in this example, equals 855.6.

#Lastly, to find the R-squared,
# the analyst takes the first sum of errors, divides it by the second sum of errors and subtracts this result from 1.
# In this example it is:

#R-squared = 1 - (91.81 / 855.6) = 1 - 0.11 = 0.8

#Read more: R-Squared http://www.investopedia.com/terms/r/r-squared.asp#ixzz4m2MW3W9l 
