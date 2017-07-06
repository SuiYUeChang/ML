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

##Lasso

#The Lasso is a linear model that estimates sparse coefficients.
# It is useful in some contexts due to its tendency to prefer solutions with fewer parameter values, 
# effectively reducing the number of variables upon which the given solution is dependent. 
# For this reason, the Lasso and its variants are fundamental to the field of compressed sensing.
# Under certain conditions, it can recover the exact set of non-zero weights 
# (see Compressive sensing: tomography reconstruction with L1 prior (Lasso)).
# Mathematically, it consists of a linear model trained with \ell_1 prior as regularizer. 
# The objective function to minimize is:
#\underset{w}{min\,} { \frac{1}{2n_{samples}} ||X w - y||_2 ^ 2 + \alpha ||w||_1}
#The lasso estimate thus solves the minimization of the least-squares penalty with \alpha ||w||_1 added, 
# where \alpha is a constant and ||w||_1 is the \ell_1-norm of the parameter vector.

from sklearn import linear_model
reg = linear_model.Lasso(alpha = 0.1)
reg.fit([[0, 0], [1, 1]], [0, 1])
#Lasso(alpha=0.1, copy_X=True, fit_intercept=True, max_iter=1000,
   #normalize=False, positive=False, precompute=False, random_state=None,
   #selection='cyclic', tol=0.0001, warm_start=False)
reg.predict([[1, 1]])
#array([ 0.8])
