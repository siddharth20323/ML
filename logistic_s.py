# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, f1_score

# Generate synthetic binary classification data
X_log = np.random.rand(100, 1) * 10
y_log = (X_log > 5).astype(int)  # Target variable: 0 or 1

# Add a bias term (x0 = 1)
X_b_log = np.c_[np.ones((X_log.shape[0], 1)), X_log]

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Gradient Descent for Logistic Regression
def logistic_regression(X, y, lr=0.1, epochs=1000):
    m = len(X)
    theta = np.random.randn(2, 1)
    for epoch in range(epochs):
        gradients = 1/m * X.T.dot(sigmoid(X.dot(theta)) - y)
        theta -= lr * gradients
    return theta

# Train model
theta_log = logistic_regression(X_b_log, y_log)

# Predict probabilities and classes
y_prob = sigmoid(X_b_log.dot(theta_log))
y_pred_log = (y_prob >= 0.5).astype(int)

# Confusion Matrix
tn, fp, fn, tp = confusion_matrix(y_log, y_pred_log).ravel()

# Metrics
accuracy = (tp + tn) / (tp + tn + fp + fn)
error_rate = 1 - accuracy
recall = tp / (tp + fn)            # Sensitivity
specificity = tn / (tn + fp)       # True Negative Rate
f1 = f1_score(y_log, y_pred_log)
auc = roc_auc_score(y_log, y_prob)

# Output
print(f"Accuracy: {accuracy:.4f}")
print(f"Error Rate: {error_rate:.4f}")
print(f"True Positives (TP): {tp}")
print(f"True Negatives (TN): {tn}")
print(f"False Positives (FP): {fp}")
print(f"False Negatives (FN): {fn}")
print(f"Recall (Sensitivity): {recall:.4f}")
print(f"Specificity: {specificity:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"AUC Score: {auc:.4f}")

# Plotting
plt.scatter(X_log, y_log, color='blue', label='Data')
plt.plot(X_log, y_prob, color='red', label='Logistic Regression')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Logistic Regression with Metrics')
plt.legend()
plt.show()

# ROC Curve
fpr, tpr, _ = roc_curve(y_log, y_prob)
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.2f})', color='green')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid(True)
plt.show()





# Accuracy: 1.0000
# Error Rate: 0.0000
# True Positives (TP): 47
# True Negatives (TN): 53
# False Positives (FP): 0
# False Negatives (FN): 0
# Recall (Sensitivity): 1.0000
# Specificity: 1.0000
# F1 Score: 1.0000
# AUC Score: 1.0000