# %%
import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix,classification_report,roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# %%
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
df = pd.read_csv(url, header=None)

# %%
x= df.iloc[:, :-1]  # Features
y = df.iloc[:, -1]  #target

# %%
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# %%
model=KNeighborsClassifier()
model.fit(x_train,y_train)

# %%
y_pred = model.predict(np.array(x_test))


# %%
accuracy=accuracy_score(y_test,y_pred)
cm=confusion_matrix(y_test,y_pred)
report=classification_report(y_test,y_pred)
roc_auc=roc_auc_score(y_test
                      ,y_pred)


print(f'Accuracy: {accuracy}')
print(f'Confusion Matrix:\n{cm}')
print(f'Classification Report:\n{report}')
print(f'ROC AUC Score: {roc_auc}')

# %%
x_array = np.ascontiguousarray(x.to_numpy())
cv_scores = cross_val_score(model, x_array, y, cv=10, scoring='accuracy')
print(f'Cross-validated Accuracy: {cv_scores.mean()}')




# Accuracy: 0.7904451682953312
# Confusion Matrix:
# [[450  81]
#  [112 278]]
# Classification Report:
#               precision    recall  f1-score   support

#            0       0.80      0.85      0.82       531
#            1       0.77      0.71      0.74       390

#     accuracy                           0.79       921
#    macro avg       0.79      0.78      0.78       921
# weighted avg       0.79      0.79      0.79       921

# ROC AUC Score: 0.7801390699695785
# Cross-validated Accuracy: 0.7883174573233991