""" 
Explore feature engineering and polynomial regression which allows you to use the machinery of linear regression
to fit very complicated, even very non-linear functions
"""

import numpy as np
import matplotlib.pyplot as plt
from lab_utils_multi import zscore_normalize_features, run_gradient_descent_feng
np.set_printoptions(precision=2)  # reduced display precision on numpy arrays

# Feature Engineering and Polynomial Regression Overview

# We've considered linear regression, but otherwise we can consider the below:

# Polynomial Features
# How we can fit a non-linear curve, starting off with a simple quadratic curve e.g. y = 1 + x^2


# create target data
x = np.arange(0, 20, 1)
y = 1 + x**2
X = x.reshape(-1, 1)

model_w,model_b = run_gradient_descent_feng(X,y,iterations=1000, alpha = 1e-2)

plt.scatter(x, y, marker='x', c='r', label="Actual Value"); plt.title("no feature engineering")
plt.plot(x,X@model_w + model_b, label="Predicted Value");  plt.xlabel("X"); plt.ylabel("y"); plt.legend(); plt.show()


# The predicted linear value isn't a good fit for the actual values. we need something like y = w0x0^2 + b or a polynomial feature
# We can modify the input data to engineer the needed features.
# To achieve the effect that we want, we can swap x with x**2 (x^2):

# create target data
x = np.arange(0, 20, 1)
y = 1 + x**2

# Engineer features 
X = x**2      #<-- added engineered feature

X = X.reshape(-1, 1)  #X should be a 2-D Matrix
model_w,model_b = run_gradient_descent_feng(X, y, iterations=10000, alpha = 1e-5)

plt.scatter(x, y, marker='x', c='r', label="Actual Value"); plt.title("Added x**2 feature")
plt.plot(x, np.dot(X,model_w) + model_b, label="Predicted Value"); plt.xlabel("x"); plt.ylabel("y"); plt.legend(); plt.show()


# Selecting Features
# If it's not obvious which features are required, you could add a variety of potential features to try and find the most useful. 
# What if we tried y = w0x0 + w1x1^2 + w2x2^3 + b instead? 

# create target data
x = np.arange(0, 20, 1)
y = x**2

# engineer features .
X = np.c_[x, x**2, x**3]   #<-- added engineered feature

model_w,model_b = run_gradient_descent_feng(X, y, iterations=10000, alpha=1e-7)

plt.scatter(x, y, marker='x', c='r', label="Actual Value"); plt.title("x, x**2, x**3 features")
plt.plot(x, X@model_w + model_b, label="Predicted Value"); plt.xlabel("x"); plt.ylabel("y"); plt.legend(); plt.show()

# it says w [0.08 0.54 0.03] and b is [0.0106] which means the model is 0.08x+0.54x^2 + 0.03x^3 + 0.0106
# gradient descent emphasized that increasing the w1 more relative to the others gives it a more accurate reading
# the weights are much higher for w1 because its the most useful in fitting the data


#Alternate Method: instead of the previous method where we chose polynomial featurse based on how well they matched the target data,
# Create new features such as x, x^2, and x^3 and the best features will be linear relative to the target
# We can still use linear regression after creating new features with this method

# create target data
x = np.arange(0, 20, 1)
y = x**2

# engineer features .
X = np.c_[x, x**2, x**3]   #<-- added engineered feature
X_features = ['x','x^2','x^3']

fig,ax=plt.subplots(1, 3, figsize=(12, 3), sharey=True)
for i in range(len(ax)):
    ax[i].scatter(X[:,i],y)
    ax[i].set_xlabel(X_features[i])
ax[0].set_ylabel("y")
plt.show()

# we can see that x^2 feature mapped against the target y is linear. Linear regression can generate a model using that feature


# Scaling features! 
# if the data set has features with significantly different scales, one should apply feature scaling to speed up gradient descent.
# in the example above, since we have x, x^2, and x^3, they will have very different scales. Let's apply Z-normalisation

# create target data
x = np.arange(0,20,1)
X = np.c_[x, x**2, x**3]
print(f"Peak to Peak range by column in Raw        X:{np.ptp(X,axis=0)}")

# add mean_normalization 
X = zscore_normalize_features(X)     
print(f"Peak to Peak range by column in Normalized X:{np.ptp(X,axis=0)}")

# trying again with a more aggresive value of alpha
x = np.arange(0,20,1)
y = x**2

X = np.c_[x, x**2, x**3]
X = zscore_normalize_features(X) 

model_w, model_b = run_gradient_descent_feng(X, y, iterations=100000, alpha=1e-1)

plt.scatter(x, y, marker='x', c='r', label="Actual Value"); plt.title("Normalized x x**2, x**3 feature")
plt.plot(x,X@model_w + model_b, label="Predicted Value"); plt.xlabel("x"); plt.ylabel("y"); plt.legend(); plt.show()
# feature scaling allows this to converge much faster. 

# Complex Functions
# with feature engineering, even complex functions can be modeled; seen in the next example 

x = np.arange(0,20,1)
y = np.cos(x/2)

X = np.c_[x, x**2, x**3,x**4, x**5, x**6, x**7, x**8, x**9, x**10, x**11, x**12, x**13]
X = zscore_normalize_features(X) 

model_w,model_b = run_gradient_descent_feng(X, y, iterations=1000000, alpha = 1e-1)

plt.scatter(x, y, marker='x', c='r', label="Actual Value"); plt.title("Normalized x x**2, x**3 feature")
plt.plot(x,X@model_w + model_b, label="Predicted Value"); plt.xlabel("x"); plt.ylabel("y"); plt.legend(); plt.show()
