from sklearn.datasets import load_iris
from sklearn import tree
from sklearn import linear_model
import numpy as np

iris = load_iris()

test_idx = [0, 10, 20, 30, 40, 50, 100, 120, 130, 145, 146]

#training data

train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx, axis = 0)


# testing data

test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

clf = linear_model.LogisticRegression(C=1e5)
clf = clf.fit(train_data, train_target)


print (test_target)
print (clf.predict(test_data))













