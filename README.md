# ML_trial
Trying out Machine Learning techniques on MNIST digit dataset

kmeans.py utilizes the k-means algorithm of unsupervised clustering
    - can choose to either standardize or normalize the data
        - standardize accuracy = ~52-54%
        - normalize accuracy = ~60%

SVC.py utilizes the linear support vector classification algorithm (supervised) with an accuracy of ~85% (before preprocessing data)
    - included option of using either using standardized data or normalized data (normalized is default)
        - standardized accuracy = ~91%
        - normalized accuracy = ~92%
        - NOTE: normalized option made algorithm run (much) faster due to smaller scale of values

keras-trial.py utilizes a Convolutional Neural Network with normalized data, achieving an accuracy of ~98-99%
    - beginning example of using Kera's API on the MNIST digit dataset
    - has a visual example of the first 50 images and the model's predictions and actual values
