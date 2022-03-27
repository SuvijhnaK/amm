import sklearn.metrics as sm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn import datasets
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

iris = datasets.load_iris()

X = pd.DataFrame(iris.data)
X.columns = ['Sepal_Length','Sepal_Width','Petal_Length','Petal_Width']
y = pd.DataFrame(iris.target)
y.columns = ['Targets']

model = KMeans(n_clusters = 3)
model.fit(X)
score1 = sm.accuracy_score(y, model.labels_)
print('Accuracy', score1,'\n')

plt.figure(figsize=(7,7))
colormap = np.array(['red','lime','black'])
plt.subplot(1,2,1)
plt.scatter(X.Petal_Length,X.Petal_Width, c = colormap[model.labels_],s = 40)
plt.title('K Means Classification')

scaler = preprocessing.StandardScaler()
scaler.fit(X)
xsa = scaler.transform(X)
xs = pd.DataFrame(xsa, columns = X.columns)

gmm = GaussianMixture(n_components=3)
gmm.fit(xs)
y_cluster_gmm = gmm.predict(xs)
score2 = sm.accuracy_score(y, y_cluster_gmm)
print('Accuracy of EM Algorithm is:', score2)

plt.subplot(1,2,2)
plt.scatter(X.Petal_Length,X.Petal_Width, c = colormap[y_cluster_gmm],s = 40)
plt.title('EM classification')
