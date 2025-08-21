import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

data={
    "shear":[2063.2, 1760.5, 713.1, 1977.4, 1645.0, 1121.3, 2101.8, 1810.7, 817.2, 1718.5,
         990.1, 2004.0, 1283.8, 1583.2, 1786.4, 1489.6, 1210.0, 2137.0, 1353.4, 1038.3,
         2035.6, 1697.1, 952.7, 1437.9, 1575.0, 1022.4, 2119.2, 1959.8, 1094.0, 1740.6],
    "age":[7.23, 12.91, 29.13, 9.33, 14.55, 21.07, 6.45, 10.94, 27.52, 13.47,
       24.65, 8.91, 19.40, 15.06, 11.26, 16.78, 20.21, 5.89, 18.30, 22.56,
       7.99, 13.88, 25.34, 17.43, 14.99, 23.18, 6.11, 9.87, 21.95, 12.39]
}
print(data.items())

df=pd.DataFrame(data)
df.head()
Y=data['shear']
X=data['age']
X=sm.add_constant(X)
linear_regression=sm.OLS(Y,X)
fitted_model=linear_regression.fit()
fitted_model.summary()
intercept=fitted_model.params[0]
slope=fitted_model.params[1]
print("\nIntercept:",intercept)
print("slope:",slope)
