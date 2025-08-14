# Chapter 3 - Classification 

## MNIST
The hello world for machine learning and classification. It is a collection of 70,000 images of handwritten numbers from high school students and employees. 

`from sklearn.datasets` - has three different types of functions `fetch_*` are used to download real-life datasets, `load_*` to load small toy datasets bundled with Scikit-learn and `make_*` functions to generate fake datasets. 

There are 70,000 images in the MNIST dataset and there are 784 features. The features are the individual pixels of the 28 x 28 frame. Each pixel is on the rbg scale from 0 to 255.

**Important to split MNIST data set into training and testing before diving deep into the dataset, this is to remove some of the personal bias**. Additionally, the dataset is already shuffled for us which is important for some algorithms, since some perform poorly if they get many of the similar instances in a row. 

## Training a Binary Classifer 
For this chapter we will be using a stochastic gradient descent which is great for classification efficiently. 

## Performance Measures 

Evaluating performance on classifiers is more tricky. 

### Measuring Accuracy Using Cross-validation 

Great way to eval a classifier is with cross-validation. 

We use k-fold cross-validation which splits the training set into k-folds (e.g split the data into k groups with no overlapping data points). Then we train the model k-times, holding out a different fold each time for validation 

One of the main reasons why accuracy is not the preferred performance metric is that a lot of classification datasets are skewed. This means that the accuracy rate may not be telling the full picture of the models performance, but a better method is to use confusion matrix (CM)

## Confusion Matrices 

Idea: counts the number of times the classifier misclassified type A with type B for all the pairs of type A/B. Funny enough it gets its name because it shows where the classifer got confused.  

Example: the classifier misclassified 9 for 7 then it would look at row 9 column 7. 

To create this matrix you first need to predictions on your training set. 

### `cross_val_predict` from `sklearn.model_selection`

This produces out-of-sample or clean predictions, meaning the models makes predictions on data that it never saw during training. How this works is through the k-folds, in the notebook we split the training data into 3 folds. For the clean prediction we do three separate iterations (train the model 3 times). Each iteration we train the model on separate folds for example in the first iteration we train on folds 1 and 2 and then the model predicts the outputs for fold 3. In this case the model never saw the instance of the data in training. We repeat the same process for the other iterations and then in the end we combine all the predicted outputs to get clean predictions or out-of-sample. 

### precision 

precision = $\frac{TP}{TP+FP}$

TP - True positives, FP - False positives 

This metric can quickly be manipulate by only classifying negative predictions and making one positive prediction you know for certain is True. This leads to 100 percent precision. So often this metric is done with other metrics 

### recall - sensitivity or true positive rate 
The ratio of positive instances that are correctly detected byt he classifier. 

recall = $\frac{TP}{TP + FN} $ 

FN - False negatives 

### F1 score 

combination of the metrics recall & precision, basically the harmonic mean between the two. This gives more weight to lower scores and thus the F1 score will only be high if both the precision and recall are super high. 

$F_1 = \frac{2}{\frac{1}{precision} + \frac{1}{recall}} = 2 \times \frac{precision \times recall}{precision + recall} = \frac{TP}{TP + \frac{FN + FP}{2}}$

NOTE: there is a trade-off between precision and recall and depending on the classifier you want to build you may favor one over the other. 


### The Precision/Recall Trade-Off 

**decision function - in layman terms computes the confidence score it has for the classification it's making on a specific item** 

First understand SGDClassifer 
computes score based on decision features -> if score falls in threshold -> add to positive class otherwise goes to negative. 

From the notebook we know that the threshold can impact the recall and precision. In the notebook if the threshold is higher then we get a lot of false negatives decreasing the recall accuracy. 

## ROC Curve (receiver operating characteristics) 

This curve plots the true positive rate (recall) versus the false positive rate. 

FPR (fall-out)= 1 - true negative rate 
TNR - true negative rate - specificity 

ROC = sensitivity (recall) versus 1 - specificity  

**ROC and PR are very similar, you should go with PR when the positive class is rare or when you care about the false postiive


## Multiclass classification 

Some scikit-learn classifiers that handle multiple classification are LogisticRegression, RandomForestClassifier and GaussianNB 

The Stochastic gradient descent from the last section is strictly binary classifier. But, you can built up Multiclass classifiers with creating a system. (e.g train 10 classifiers to classify 0-9 digits). From there it would compute the decision score and then compare it to the closest Classifier. This is OvR (one verses the rest) or OvA (one verses all)

Another approach  is OvO (one versus one), this is training a pair on each digit, one distinguishes 0 and 1, 0 and 2, 1 and 2, etc. This trains N X (N-1)/2 classifiers but only trains on the subset of the training data (e.g classifier would train only on images of 0 and 1). 

## Error Analysis 

* 1st look at confusion matrix, first use cross_val_predict(), then run it on confusion_matrix() and plot the results to see the errors. 

* once you have the confusion matrix plotted you can quickly see what labels are being misclassified and which label they are being misclassified too. Additionally, you can play around with the plot to only see the errors and the percentages (e.g exclude the correct predictions). This allows you to see where your classifier my be going wrong and points were you could fix it 

- For error fixing, you can focus on gather more data to allow your classifier to train on more data. This could be a great time to introduce new data on specific label that looks similar to another to build a better classifier. Additionally, you can engineer new features like focusing on certain parts of the images to improve classification. Lastly, you can just analyze individual errors to improve your classifier gradually. 

## Multilabel classification 

Not every classifier we make we will want to make predictions and produce one label. Sometimes we want it to produce multiple labels for a single image or item. A quick example would be face recognition where we would want to label each person in an image rather than just one. 

## Multioutput classification 

Generalization of multilabel classification where each outputs can be associated to multiple classes (e.g contain multiple values) 

