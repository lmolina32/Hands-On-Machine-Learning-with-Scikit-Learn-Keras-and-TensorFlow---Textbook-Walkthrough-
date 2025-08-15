# Training Models - chapter 4 

You can do a lot of ml without knowing the underlying implementations. But, knowing what happens under to hood will allow you to pick better models quicker, tune models faster and perform analysis better. 

This chapter focuses on linear regression with two approaches 'closed-form equations' and then iterative optimization with different forms of gradient descent. 

Additional things covered in this chapter: 
- polynomial regression 
- learning curves 
- reducing overfitting 
- regularization techniques 
- logistic & softmax regression 

## Linear Regression 

Generally, a linear model makes a prediction by simply computing a weighted sum of the input features, plus a constant called the bias term. 

$ y = \theta_0 + \theta_1x_1 + \theta_2x_2 + \dots + \theta_nx_n$

- y -> predicted value  
- n -> number of features 
- $x_i$ -> ith feature value 
- $\theta_I$ -> jth model parameter 


$ y = h_{\theta}(x)=\theta * x$
- $h_{\theta}$ -> hypothesis fucntion 
- $\theta$ models parameter vector 
- x instance's feature vector 

### how do you train Linear regression? 

Fit the model with the hyperparameter then minimize the cost function (e.g RMSE or MSE) 

* Note: often you use different loss function for training and then use another performance measure to evaluate the model. This is because the function is easier to optimize or it has to account for additional hyper-parameters 

## The Normal Equation 

The normal equation is a closed form equation that gives the the $\theta$ that minimizes the MSE. 

$\hat{\theta}=(X^TX)^{-1} X^Ty$

* $\hat{\theta}$ - contains the $\theta$ that minimizes the cost function 
* y is the vector of the target values 

## Computational Complexity 

The normal equation time complexity is about O(n^2.4) to O(n^3) where as singular value decomposition (SVD) is about O(n^2). This is still not very good and it will slow down if the number of features in a dataset is extremely large. But, since they are both linear they will handle the large training sets efficiently given that you have enough memory. 

## Gradient Descent