# %%
import numpy as np
import pandas as pd 
from sklearn.linear_model import Lasso,Ridge
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error,r2_score

# %%
data=pd.read_csv("/Users/siddharthkumar2023/Downloads/student+performance/student/student-mat.csv",sep=';')

# %%
data.head()

# %%

x= data[['studytime', 'failures', 'absences']]

y=data['G3']

# %%


# %%
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# %%
#lasso

lasso_m=Lasso(alpha=0.1)
lasso_m.fit(x_train,y_train)

# %%
y_pred_lasso=lasso_m.predict(x_test)


# %%
mse_lasso=mean_squared_error(y_test,y_pred_lasso)
r2_lasso=r2_score(y_test,y_pred_lasso)

print(f'Lasso Regression - MSE: {mse_lasso}, R²: {r2_lasso}')


# %%

cv_scores_lasso = cross_val_score(lasso_m, x, y, cv=10, scoring='neg_mean_squared_error')
print(f'Lasso Regression - Cross-validated MSE: {-cv_scores_lasso.mean()}')


# %%
#ridge
ridge_m=Ridge(alpha=0.1)
ridge_m.fit(x_train,y_train)

# %%
y_pred_ridge=ridge_m.predict(x_test)

# %%
mse_ridge=mean_squared_error(y_test,y_pred_ridge)
r2_ridge=r2_score(y_test,y_pred_ridge)

print(f'ridge Regression - MSE: {mse_ridge}, R²: {r2_ridge}')


# %%
cv_scores_ridge = cross_val_score(ridge_m, x, y, cv=10, scoring='neg_mean_squared_error')
print(f'ridge Regression - Cross-validated MSE: {-cv_scores_ridge.mean()}')

# %%


# Lasso Regression - MSE: 19.525327905243405, R²: 0.0477794247442872
# Lasso Regression - Cross-validated MSE: 18.688081416720802

# ridge Regression - MSE: 19.873632117655, R²: 0.030793157516606384
# ridge Regression - Cross-validated MSE: 18.68858708540078
 



