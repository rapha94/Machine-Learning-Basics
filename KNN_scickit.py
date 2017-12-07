from sklearn.datasets import *
import pandas as pd
iris = load_iris()
ir = pd.DataFrame(iris.data)
ir.columns = iris.feature_names
ir['CLASS'] = iris.target
ir.head()

from sklearn.neighbors import NearestNeighbors


nn = NearestNeighbors(5)
nn.fit(iris.data)


import numpy as np
test =  np.array([5.3,3,2,2.5])
test1 = test.reshape(1,-1)

nn.kneighbors(test1,5)
print(ir.ix[[98, 64, 43, 23, 57],])


