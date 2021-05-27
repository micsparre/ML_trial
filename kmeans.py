from keras.datasets import mnist
from sklearn.cluster import KMeans
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import mode
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn import metrics
import seaborn as sns;

# loading the MNIST digit dataset
(train_X, train_y), (test_X, test_y) = mnist.load_data()

#shape of dataset
print('X_train: ' + str(train_X.shape))
print('Y_train: ' + str(train_y.shape))
print('X_test:  '  + str(test_X.shape))
print('Y_test:  '  + str(test_y.shape))

# reshape the training and test data so it can be passed into the learning algorithm
train_X = train_X.ravel()
train_X = train_X.reshape((60000, 784))
test_X = test_X.ravel()
test_X = test_X.reshape((10000, 784))

## CHOOSE either standardized or normalized data

# standardize data
# scaler = StandardScaler()
# scaler.fit(train_X)
# train_X_scaled = scaler.transform(train_X)
# test_X_scaled = scaler.transform(test_X)

# normalize data
normalizer = Normalizer()
normalizer.fit(train_X)
train_X_scaled = normalizer.transform(train_X)
test_X_scaled = normalizer.transform(test_X)

# run the kmeans algorithm on the digit training set
k_means = KMeans(n_clusters = 10, init = 'random', n_init = 20)
k_means.fit(train_X_scaled)

# setup figure and plot the clusters for visual performance check
fig, ax = plt.subplots(2, 5, figsize=(8, 3))
# centers = scaler.inverse_transform(k_means.cluster_centers_).reshape(10, 28, 28) # standardize option
centers = k_means.cluster_centers_.reshape(10, 28, 28)  # normalize option
for axi, center in zip(ax.flat, centers):
    axi.set(xticks=[], yticks=[])
    axi.imshow(center, interpolation='nearest', cmap=plt.cm.binary)
plt.show(block=True)


# compute the clusters on test data and assign labels to them
clusters = k_means.predict(test_X_scaled)
labels = np.zeros_like(clusters)
for i in range(10):
    mask = (clusters == i)
    labels[mask] = mode(test_y[mask])[0]

# check accuracy of test data on model from training data
print(accuracy_score(test_y, labels))
plot2 = plt.figure(2)
disp = metrics.confusion_matrix(test_y, labels)
sns.heatmap(disp.T, square=True, annot=True, fmt='d', cbar=False)
plt.xlabel('true label')
plt.ylabel('predicted label');
plt.show(block=True)
