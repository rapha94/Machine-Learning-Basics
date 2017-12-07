from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import logisticRegression
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt


cancer = load_brest_cancer()


X_train, X_test, Y_train, Y_test = train_test_split(cancer.data, cancer.target, stratify =cancer.target, random_state = 42)


log_reg = logisticRegression()
log_reg.fit(X_train, Y_train)














