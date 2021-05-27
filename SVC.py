from keras.datasets import mnist
from sklearn import svm, metrics
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler, Normalizer

# loading the MNIST digit dataset
(train_X, train_y), (test_X, test_y) = mnist.load_data()

# shape of dataset
print('X_train: ' + str(train_X.shape))
print('Y_train: ' + str(train_y.shape))
print('X_test:  '  + str(test_X.shape))
print('Y_test:  '  + str(test_y.shape))

# flatten data
n_samples = len(train_X)
train_X = train_X.reshape((n_samples, -1))
test_X = test_X.reshape((len(test_X), -1))

# build a support vector classifier (WARNING: time complexity of alg makes next line take a couple minutes)
classifier = svm.LinearSVC()
# classifier = svm.SVC(kernel = 'linear') # gives increased accuracy at cost of more time

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

# learn the digits on the training data
classifier.fit(train_X_scaled, train_y)

#predict test data labels
predicted = classifier.predict(test_X_scaled)

# plot first 6 test samples and their predictions
_, axes = plt.subplots(nrows=1, ncols=10, figsize=(10, 3))
for ax, image, prediction in zip(axes, test_X, predicted):
    ax.set_axis_off()
    image = image.reshape(28, 28)
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    ax.set_title(f'Prediction: {prediction}')

# output the confusion matrix
disp = metrics.plot_confusion_matrix(classifier, test_X_scaled, test_y, normalize = 'true')
disp.figure_.suptitle("Confusion Matrix")
# print(f"Confusion matrix:\n{disp.confusion_matrix}")

# output the classification report
print(f"Classification report for classifier {classifier}:\n"
      f"{metrics.classification_report(test_y, predicted)}\n")

plt.show(block=True) # necessary to show plots when ran in terminal
