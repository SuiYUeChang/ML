# coding = 'utf-8'
# this is about scikit-learn


# Ordinary Least Squares
#\underset{w}{min\,} {|| X w - y||_2}^2
from sklearn import linear_model
reg = linear_model.LinearRegression()
reg.fit ([[0, 0], [1, 1], [2, 2]], [0, 1, 2])
#LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
reg.coef_
#array([ 0.5,  0.5])

#multicollinearity
#
#However, coefficient estimates for Ordinary Least Squares rely on the independence of the model terms.
#When terms are correlated and the columns of the design matrix X have an approximate linear dependence,
#the design matrix becomes close to singular and as a result,
#the least-squares estimate becomes highly sensitive to random errors in the observed response,producing a large variance.
#This situation of multicollinearity can arise, for example, when data are collected without an experimental design.

#Ridge Regression
#\underset{w}{min\,} {{|| X w - y||_2}^2 + \alpha {||w||_2}^2}

from sklearn import linear_model
reg = linear_model.Ridge (alpha = .5)
reg.fit ([[0, 0], [0, 0], [1, 1]], [0, .1, 1]) 
#Ridge(alpha=0.5, copy_X=True, fit_intercept=True, max_iter=None,
#      normalize=False, random_state=None, solver='auto', tol=0.001)
reg.coef_
#array([ 0.34545455,  0.34545455])
reg.intercept_ 
#0.13636...

## Setting the regularization parameter: generalized Cross-Validation
from sklearn import linear_model
reg = linear_model.RidgeCV(alphas=[0.1, 1.0, 10.0])
reg.fit([[0, 0], [0, 0], [1, 1]], [0, .1, 1])       
#RidgeCV(alphas=[0.1, 1.0, 10.0], cv=None, fit_intercept=True, scoring=None,
  #  normalize=False)
reg.alpha_                                      
#0.1

