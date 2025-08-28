#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import statsmodels.api as sm


data = {
    "shear": [2160.70,1680.15,2318.00,2063.30,2209.50,1710.30,1786.70,2577.00,2359.90,2258.70,2167.20,2401.55,1781.80,2338.75,1767.30,2055.50,2416.40,2202.50,2656.20,1755.70],
    "age": [15.50, 23.75, 8.00, 17.00, 5.50, 19.00, 24.00, 2.50,7.50,11.00,13.00,3.75,25.00,9.75,22.00,18.00,6.00,12.50,2.00,21.50]
}

df = pd.DataFrame(data)


y = df['shear']
X = df['age']


X = sm.add_constant(X)


linear_regression = sm.OLS(y, X)
fitted_model = linear_regression.fit()


print(fitted_model.summary())


intercept = fitted_model.params['const']
slope = fitted_model.params['age']

print("\nIntercept:", intercept)
print("Slope:", slope)


# In[7]:


import numpy as np
def gradient_descent(x,y,initial_learning_rate=0.01, decay_rate=0.01,n_iterations=1000):
    m=len(y)
    
    theta=np.random.randn(2)
    
    for iteration in range(n_iterations):
        gradients=(2/m)*X.T.dot(X.dot(theta)-y)
        learing_rate=initial_learning_rate / (1+decay_rate*iteration)
        theta-=learing_rate*gradients
        error=np.mean((X.dot(theta)-y)**2)
        
        if error > 0.1:
            iteration+=1
        break
    return gradients
    return theta
theta_gd=gradient_descent(X,y)
print("\nGradient Descent")
print(f"Intercept:{theta_gd[0]},slope:{theta_gd[1]}")


# In[8]:


def stochastic_gradient_descent(X,y,learning_rate = 0.001,n_iterations = 1000):
    m = len(y)
    theta = np.array([3000,-0.1])
    for iteration in range(n_iterations):
        for i in range(m):
            random_index = np.random.randint(m)
            xi = X[random_index:random_index+1]
            yi = y[random_index:random_index+1]
            gradients = (2/m)*xi.T.dot(xi.dot(theta)-yi)
            theta -= learning_rate * gradients
            
        return gradients
        return theta
theta_sgd = stochastic_gradient_descent(X,y)
print("\nStochastic gradient descent: ")
print(f"Intercept: {theta_sgd[0]},slope: {theta_sgd[1]}")


# In[9]:


from sklearn.linear_model import SGDRegressor 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test= train_test_split(X,y, test_size=0.2, random_state=42)

sgd_reg= SGDRegressor(max_iter=1000, tol=1e-3)
sgd_reg.fit(X_train, y_train)

x_values=np.linspace(0,25,100)
y_pred=sgd_reg.predict(X)
print("Predictions:", y_pred)


# In[ ]:




