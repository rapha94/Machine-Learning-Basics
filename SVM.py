import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import svm


digits = datasets.load_digits()


clf = svm.SVC(gamma= 0.001, C = 100)

x,y = digits.data[:-1], digits.target[:-1]

clf.fit(x,y)
 
print(len(digits.data))

x = -4

print("Prediction of last:",clf.predict(digits.data[[x]]))

#modélie une image du chiffre (ex ci-dessous ==> 9)
plt.imshow(digits.images[x], cmap = plt.cm.gray_r, interpolation = "nearest")
plt.show()






