# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import Required Libraries

2. Load and View the Dataset

3. Drop Unnecessary Columns

4. Convert Categorical Columns to category Type

5. Convert Categories to Numeric Codes

6. Separate Features (X) and Target (y)

7. Initialize Model Parameters (Weights)

8. Define the Sigmoid Function

9. Define the Loss (Cost) Function

10. Implement Gradient Descent

11. Define Prediction Function

12. Make Predictions and Compute Accuracy

13. Predict for New Students

## Program:
```
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by:SHRAVANI M
RegisterNumber:212224230263
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
iris=load_iris()
df=pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target']=iris.target
print(df.head())
X = df.drop('target',axis=1)
y=df['target']
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)
sgd_clf=SGDClassifier(max_iter=1000, tol=1e-3)
sgd_clf.fit(X_train,y_train)
y_pred=sgd_clf.predict(X_test)
accuracy=accuracy_score(y_test,y_pred)
print(f"Accuracy: {accuracy:.3f}")
cm=confusion_matrix(y_test,y_pred)
print("Confusion Matrix:")
print(cm)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, cmap="Blues", fmt='d', xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, ConfusionMatrixDisplay, RocCurveDisplay
)
from sklearn.calibration import CalibratedClassifierCV  # optional for probabilities
import warnings
warnings.filterwarnings("ignore")
# --------------------
# 1. Load dataset
# --------------------
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)  # 0 = malignant, 1 = benign
print("Dataset:", data.DESCR.splitlines()[0])
print("X shape:", X.shape, "y shape:", y.shape)
print()
# --------------------
# 2. Quick EDA (head)
# --------------------
print("First 5 rows:")
print(X.iloc[:, :6].head())  # show a few columns only
# --------------------
# 3. Select 2 features for plotting decision boundary (optional)
#    and keep full feature set for training
# --------------------
# Two features chosen (for visual decision boundary)
feat1, feat2 = "mean radius", "mean texture"
X_vis = X[[feat1, feat2]].values

# Full features for model training
X_full = X.values
# --------------------
# 4. Train-test split
# --------------------
X_train, X_test, y_train, y_test = train_test_split(X_full, y, test_size=0.25, random_state=42, stratify=y)
Xv_train, Xv_test, yv_train, yv_test = train_test_split(X_vis, y, test_size=0.25, random_state=42, stratify=y)  # for viz
# --------------------
# 5. Scaling
# --------------------
scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)

scaler_vis = StandardScaler().fit(Xv_train)
Xv_train_s = scaler_vis.transform(Xv_train)
Xv_test_s = scaler_vis.transform(Xv_test)
# --------------------
# 6. Train SGDClassifier as logistic regression
#    - loss='log' or 'log_loss' depending on sklearn version
#    - alpha -> L2 regularization strength
# --------------------
clf = SGDClassifier(
    loss='log_loss',           
    penalty='l2',
    alpha=1e-4,
    max_iter=1000,
    tol=1e-4,
    learning_rate='optimal',  # 'constant', 'invscaling', 'optimal'
    random_state=42,
    verbose=0
)

clf.fit(X_train_s, y_train)

# SGDClassifier does not implement predict_proba by default.
# We can wrap it with CalibratedClassifierCV if we need probabilities (for ROC).
calibrated = CalibratedClassifierCV(clf, method="sigmoid", cv=5)
calibrated.fit(X_train_s, y_train)
# --------------------
# 7. Evaluation on test set
# --------------------
y_pred = clf.predict(X_test_s)
y_proba = calibrated.predict_proba(X_test_s)[:, 1]  # probability of class 1

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Test set metrics:")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1-score:  {f1:.4f}")
print()
# Cross-validation (optional)
cv_scores = cross_val_score(clf, scaler.transform(X_full), y, cv=5, scoring='accuracy')
print("5-fold CV accuracy: mean={:.4f} std={:.4f}".format(cv_scores.mean(), cv_scores.std()))
print()
# --------------------
# 8. Confusion matrix
# --------------------
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=data.target_names)
fig, ax = plt.subplots(figsize=(5,4))
disp.plot(ax=ax)
ax.set_title("Confusion Matrix (Test set)")
plt.show()
# --------------------
# 9. ROC curve
# --------------------
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)
fig, ax = plt.subplots(figsize=(6,5))
RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name="SGD Logistic (calibrated)").plot(ax=ax)
ax.set_title(f"ROC Curve (AUC = {roc_auc:.3f})")
plt.show()

# --------------------
# 10. Decision boundary plot (2 features) for visual learners
#     We'll train a separate SGD model on just the 2 features.
# --------------------
clf_vis = SGDClassifier(loss='log_loss', max_iter=1000, tol=1e-4, random_state=42)
clf_vis.fit(Xv_train_s, yv_train)

# Make mesh
xx_min, xx_max = Xv_train_s[:, 0].min() - 1, Xv_train_s[:, 0].max() + 1
yy_min, yy_max = Xv_train_s[:, 1].min() - 1, Xv_train_s[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(xx_min, xx_max, 300), np.linspace(yy_min, yy_max, 300))
grid = np.c_[xx.ravel(), yy.ravel()]

Z = clf_vis.predict(grid).reshape(xx.shape)

fig, ax = plt.subplots(figsize=(7,6))
ax.contourf(xx, yy, Z, alpha=0.2)
# plot training points
ax.scatter(Xv_train_s[:, 0][yv_train==0], Xv_train_s[:, 1][yv_train==0], marker='o', label=data.target_names[0], edgecolor='k')
ax.scatter(Xv_train_s[:, 0][yv_train==1], Xv_train_s[:, 1][yv_train==1], marker='^', label=data.target_names[1], edgecolor='k')
ax.set_xlabel(feat1 + " (scaled)")
ax.set_ylabel(feat2 + " (scaled)")
ax.set_title("Decision boundary (SGD Logistic) — trained on 2 features")
ax.legend()
plt.show()  
*/
```
## Output:
<img width="662" height="708" alt="image" src="https://github.com/user-attachments/assets/db09a62b-e4ea-433a-8bf9-f15341ff70af" />
<img width="375" height="72" alt="image" src="https://github.com/user-attachments/assets/b1cf5f7b-3609-4cd0-a48d-7dba55c2c122" />
<img width="666" height="257" alt="image" src="https://github.com/user-attachments/assets/fc6651f2-c93d-4d45-8660-29bdf7dfd304" />
<img width="352" height="176" alt="image" src="https://github.com/user-attachments/assets/b5b399f1-10f0-48c2-a4bb-b740de3a375e" />
<img width="250" height="112" alt="image" src="https://github.com/user-attachments/assets/78780aa7-f3e3-4ba8-a513-8c764b320359" />
<img width="517" height="51" alt="image" src="https://github.com/user-attachments/assets/b1f89fc2-69fa-4fff-86b7-05aeacfe1017" />
<img width="580" height="407" alt="image" src="https://github.com/user-attachments/assets/342f283a-7591-43d7-b11d-cf7369c66c6b" />
<img width="682" height="487" alt="image" src="https://github.com/user-attachments/assets/054d2fa3-48d7-450e-9cac-e82a7a24a094" />
<img width="747" height="567" alt="image" src="https://github.com/user-attachments/assets/fecf15ce-6414-4cc6-bfec-a040e5e37a67" />




## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
