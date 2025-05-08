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


# %%
x=data[['studytime','G1','G2']]
y=data['G3']

# %%
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

# %%
model=LinearRegression()
model.fit(x_train,y_train)

# %%
y_pred=model.predict(x_test)

# %%
mse=mean_squared_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)
print(f"MSE is : {mse}, r2 score is :{r2} ")

# %%

cv_scores = cross_val_score(model, x, y, cv=10, scoring='neg_mean_squared_error')
print(f'Cross-validated MSE: {-cv_scores.mean()}')



# %%


# %%
import matplotlib.pyplot as plt

# %%
plt.scatter(y_test,y_pred)
plt.xlabel("Actual grades")
plt.ylabel("predicted grades")
plt.title("ACTUAL VS PREDICTED")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# MSE is : 4.166563708144214, r2 score is :0.7968030186093205 
#  Cross-validated MSE: 3.8393132522651165
