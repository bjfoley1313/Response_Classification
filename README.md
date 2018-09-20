# Response_Classification

The first step in this model is to reduce the dimensionality of the data.  There is bound to be a large amount of collinearity among 16,500 binary features so I decided a PCA would be an effective way of reducing the number of features.  I wanted to retain the vast majority of the variance in the original data so I determined 300 components which would retain around 90% of the variation would suffice.  After creating the PCA I then fit and transformed the model to the original data.  I decided to leave the reponse class unbalanced and note that a baseline predition accuracy of 77% could be achieved by simply predicting all responses as 0.  

There was no need to scale the features or transform any of them because they were all confimed to be binary (0,1).  The only preprocessing was to transform the X_train and X_test datasets to the PCA model, reducing the number of columns in those datasets from 16,500 to 300.  

Next a Naive Bayes classification model was fit to the training data.  Since Naive Bayes has no hyperparameters it was not included in the GridSearchCV.  The Naive Bayes did not perform very well with only a 79% accuracy for test set predictions.  

Next I created a function to determine a baseline accuracy for 4 different classification models using only default hyperparameters.  Those scores give us the idea that a Support Vector Machine might be the best generalizer of our data.  

Dictionaries containing different hyperparameters were then created for each classification model.  After iterating through each combination of hyperparameters using GridSearchCV,  each model was optimized for the highest 'accuracy' score. The logistic regression, SVM Polynomial, and SVM RBF models all generated an accuracy of around 85%.  Because of this they were all examined further to determine the exact nature of the models correct and incorrect classifications.  

The three finalist models all have similar accuracy scores as well as confusion matrices.  Because SVM models tend to perform better in a high dimensional space, I believe this would be the preferred model over a Logistic Regression model.  A SVM model would also be less prone to overfit the training data when compared to a Logistic Regression model.  The best performing between SVM models was the Polynomial model with a degree of 1.  This indicates that a SVM linear model would possbily be an appropriate model for this data.  
