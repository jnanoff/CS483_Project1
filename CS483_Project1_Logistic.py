
# coding: utf-8

# In[ ]:

#Logistic Regression experimentation
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics


X = np.load("predictors.npy")
y1 = np.load("labels_color.npy")
y2 = np.load("labels_quality.npy")
y3 = np.load("labels_quality_binary.npy")

y1 = np.ravel(y1)

# instantiate a logistic regression model, and fit with X and y
model1 = LogisticRegression()
model2 = LogisticRegression()

X1_train, X1_test, y1_train, y1_test = train_test_split(X, y1, test_size=0.3, random_state=0)
X2_train, X2_test, y3_train, y3_test = train_test_split(X, y3, test_size=0.3, random_state=0)
model1 = model1.fit(X1_train, y1_train)
model2 = model2.fit(X2_train, y3_train)

predicted1 = model1.predict(X1_test)
predicted2 = model2.predict(X2_test)
probs1 = model1.predict_proba(X1_test)
probs2 = model2.predict_proba(X2_test)


print(metrics.accuracy_score(y1_test, predicted1))
print(metrics.roc_auc_score(y1_test, probs1[:, 1]))

print(metrics.accuracy_score(y3_test, predicted2))
print(metrics.roc_auc_score(y3_test, probs2[:, 1]))



# check the accuracy on the training set
#print(model1.score(X, y1))
#print(y1.mean())
#print(model2.score(X, y3))
#print(y3.mean())

