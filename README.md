### Overview
We are given a dataset that consists of Electrocardiogram (ECG) signals of single heartbeats measured at 187 consecutive time points, and associated labels describing whether the heartbeat is normal or abnormal. The aim is training a classification model that predicts whether the heartbeat is normal or abnormal using the ECG signals available.

### Methods
Our candidate classification models include penalized logistic regression, K nearest neighbors, single classification tree and random forest. We split the data into training set and test set and use 10-fold cross-validation to train each model. Prediction accuracy is evaluated on the test set. 

### Result
The test prediction accuracy for penalized logistic regression, K nearest neighbors, single classification tree and random forest is 0.834, 0.945, 0.802 and 0.968. Hence, random forest is selected as the optimal model and is used to make prediction for a new test dataset with 2000 observations. The prediction accuracy is 0.941.

