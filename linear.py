# %%
import numpy as np
import pandas as pd 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error,r2_score


# %%
data=pd.read_csv("/Users/siddharthkumar2023/Downloads/student+performance/student/student-mat.csv",sep=';')

# %%
print(data.head)

# %%
print(data.columns.tolist())
# data.columns = data.columns.str.strip()

# %%
x=data[['G1']].values
y=data['G3'].values

# %%
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

# %%
model=LinearRegression()
model.fit(x_train,y_train)

# %%
import matplotlib.pyplot as plt

# %%
y_pred=model.predict(x_test)

# %%
mse=mean_squared_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)
print(f"mse is: {mse}")
print(f"r2 is: {r2}")


# %%
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

cv_mse_scores = -cross_val_score(model, x, y, cv=kfold, scoring='neg_mean_squared_error')
cv_r2_scores = cross_val_score(model, x, y, cv=kfold, scoring='r2')

# %%
print(f"Cross-validated MSE: {np.mean(cv_mse_scores)}")
print(f"Cross-validated R² Score: {np.mean(cv_r2_scores)}")

# %%
sorted_idx = x_test[:, 0].argsort()
x_test_sorted = x_test[sorted_idx]
y_pred_sorted = y_pred[sorted_idx]

# %%
plt.figure(figsize=(8, 5))
plt.scatter(x_test, y_test, color='blue', label='Actual Data')
plt.plot(x_test_sorted, y_pred_sorted, color='red', linewidth=2, label='Regression Line')
plt.xlabel('G1 (First Period Grade)')
plt.ylabel('G3 (Final Grade)')
plt.title('Simple Linear Regression: G1 vs G3')

x = np.linspace(min(y_test), max(y_test), 100)
plt.plot(x, x, color='red')

plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


