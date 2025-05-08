# %%
import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix,classification_report,roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns


# %% [markdown]
# 

# %%
# data=pd.read_csv("/Users/siddharthkumar2023/Downloads/student+performance/student/student-mat.csv",sep=';')

# %%
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
df = pd.read_csv(url, header=None)

# %%
x = df.iloc[:, :-1]  # Features
y = df.iloc[:, -1]  #target

# %%
# x = data[['studytime', 'failures', 'absences']]
# y = (data['G3'] >= 10).astype(int)  # Binary target variable: 1 for pass, 0 for fail


# %%
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# %%
model = LogisticRegression(max_iter=1000)

model.fit(x_train,y_train)

# %%
y_pred=model.predict(x_test)

# %%
#evaluation
accuracy=accuracy_score(y_test,y_pred)
cm=confusion_matrix(y_test,y_pred)
report=classification_report(y_test,y_pred)
roc_auc=roc_auc_score(y_test,y_pred)


print(f'Accuracy: {accuracy}')
print(f'Confusion Matrix:\n{cm}')
print(f'Classification Report:\n{report}')
print(f'ROC AUC Score: {roc_auc}')



# %%
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Use a pipeline to scale then fit the model
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
model.fit(x_train, y_train)

# %%
cv_score=cross_val_score(model,x,y,cv=10,scoring='accuracy')
print(f'Cross-validated Accuracy: {cv_score.mean()}')

# %%
sns.heatmap(cm,annot=True,fmt='d',cmap='Blues')
plt.xlabel("predicted")
plt.ylabel("actual")
plt.show()


# %%




# Accuracy: 0.9207383279044516
# Confusion Matrix:
# [[505  26]
#  [ 47 343]]
# Classification Report:
#               precision    recall  f1-score   support

#            0       0.91      0.95      0.93       531
#            1       0.93      0.88      0.90       390

#     accuracy                           0.92       921
#    macro avg       0.92      0.92      0.92       921
# weighted avg       0.92      0.92      0.92       921

# ROC AUC Score: 0.9152614805157179
# Cross-validated Accuracy: 0.9169716118079789