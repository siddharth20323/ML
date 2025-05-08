# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# %%
data=pd.read_csv("/Users/siddharthkumar2023/Downloads/student+performance/student/student-mat.csv",sep=';')

# %%
X=data[['G1']].values
y=data['G3'].values 

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %%
degree = 2  # You can change the degree for higher-order polynomials
poly = PolynomialFeatures(degree=degree)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# %%
model = LinearRegression()
model.fit(X_train_poly, y_train)

# %%
y_pred = model.predict(X_test_poly)

# %%
mse=mean_squared_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)
print(f"mse is: {mse}")
print(f"r2 is: {r2}")


# %%
plt.scatter(X, y, color='blue', label='Actual Data')
X_plot = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
X_plot_poly = poly.transform(X_plot)
y_plot = model.predict(X_plot_poly)
plt.plot(X_plot, y_plot, color='red', label='Polynomial Regression Fit')
plt.title(f'Polynomial Regression (Degree {degree})')
plt.xlabel('First Period Grade (G1)')
plt.ylabel('Final Grade (G3)')
plt.legend()
plt.grid(True)
plt.show()


 
# mse is: 6.174951427786729
# r2 is: 0.6988570010563485
# Cross-validated MSE: 7.598836361468244
# Cross-validated R² Score: 0.6308421989644717

