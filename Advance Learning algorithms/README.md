## Day 29

### 1- Advance Learning Algorithms Coursera by Andrew Ng

#### Forward propagation in NumPy

$$ a_1^{[1]} = g(w_1^{[1]} * x + b_1^{[1]}) $$

In python:

```
w = np.array([[1, -3, 5]
              [2, 4, -6]])
              
b = np.array([-1, 1, 2])

a_in = np.array([-2, 4]) # could be a_0 which is equal to x

def dense(a_in, W, b, g):
    
    units = W.shape[1] # columns of the w matrix = 3
    a_out = np.zeros(units) # [0,0,0]
    
    for j in range(units): # 0,1,2
        w = W[:,j]
        z = np.dot(w, a_in) + b[j]
        a_out[j] = g(z)
        
    return a_out
    
def sequential(x):

    a1 = dense(x, W1, b1)
    a2 = dense(a1, W2, b2)
    a3 = dense(a2, W3, b3)
    a4 = dense(a3, W4, b4)
    f_x = a4
    return f_x
    

```

#### Vector Matrix Multiplication

Vectors can be represented as transpose vector, in which from column vector we can have a row vector and the results will be the same

$$
a = 
\begin{pmatrix}
1\\ 
2\\
\end{pmatrix}
$$

Is equal to:

$$
a^T = 
\begin{pmatrix}
1 & 2\\
\end{pmatrix}
$$

We dfine the W matrix

$$
W = 
\begin{pmatrix}
3 & 5\\ 
4 & 6\\
\end{pmatrix}
$$

$$ Z = a^TW $$

$$ (1 * 3) + (2 * 4) = 3 + 8 = 11 $$

$$ (1 * 5) + (2 * 6) = 5 + 12 = 17 $$

$$ Z = [11  17] $$


#### Matrix Matrix Multiplication

Matrix transpose takes the columns and converted into rows

$$
A = 
\begin{pmatrix}
1 & -1\\ 
2 & -2\\
\end{pmatrix}
$$

$$
A^T = 
\begin{pmatrix}
1 & 2\\ 
-1 & -2\\
\end{pmatrix}
$$

$$
W = 
\begin{pmatrix}
3 & 5\\ 
4 & 6\\
\end{pmatrix}
$$


$$ Z = A^TW $$

$$
Z = 
\begin{pmatrix}
a_1^TW_1 & a_1^TW_2\\ 
a_2^TW_1 & a_2^TW_2\\
\end{pmatrix}
$$



row1 col1

row2 col1

row1 col2

row2 col2

**Vectorize Implementation**

```
AT = np.array([[200, 17]])

W = np.array([[1, -3, 5],
              [-2, 4, -6])

b = np.array([[-1, 1, 2]])

def dense(AT, W, b, g):
    
    z = np.matmul(AT, W) + b
    a_out = g(z)
    
    return a_out

```


## Day 30
### 1- Advance Learning Algorithms by Andrew Ng

### Model training Steps

 1. Specify how to compute output given input X and parameters w and b 
 2. Specify the loss and cost
 3. Train on data to minimize the cost function

Logistic regression steps

1)

    z = np.dot(w, x) + b
    f_x = 1/(1+np.exp(-z))

2)

    loss = -y * np.log(f_x) -(1-y) * np.log(1-f_x)
 3)

    w = w - alpha * dj_dw
    b = b - alpha * dj_db

Neural Network steps

1)

    import tensorflow as tf
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Dense
    model = sequential([
		    Dense(units = 25, activation = "sigmoid")
		    Dense(units=15, activation = "sigmoid")
		    Dense(units=1, activation = "sigmoid")
				    ])
2) 

    # binary prediction
    from tensorflow.keras.losses import BinaryCrossentropy 
    model.compile(loss = BinaryCrossentropy())
    
    # regression prediction
    from tensorflow.keras.losses import MeanSquaredError 
    model.compile(loss = MeanSquaredError())


3) 

    # Minimize the cost function using back propagation
    model.fit(X, y, epochs=100)


## Day 31

### 1- Advanced Learning Algorithms by Andrew Ng

Activation Functions

There are three activations functions that we can use in many ways to came up with powerful models:

Let's breakdown by layers:

***Output layer***

**Linear activation functions**: Better for values that can take positive and negative values
**Sigmoid functions**: Better suit for binary classification problems
**ReLU**: Better for positive numbers such as house price

***Hidden layer***
ReLU is the most common activation function for the hidden layers since it provide a faster learning. The explanation behind this statement is that ReLU is flat on just one side of the graph while sigmoid is flat in both places which make gradient descent converge in a slower pace. 

![activation_functions](https://user-images.githubusercontent.com/46135649/183745294-19642389-7eb5-43c1-a663-8b1407072573.png)

The code implementation of the above theory is:
```python
from tf.keras.layers import Dense
model = Sequential([
	Dense(units=25, activation='relu'),
	Dense(units=15, activation='relu'),
	# for binary output
	Dense(units=1, activation='sigmoid') 
	])
```

## Day 32

### 1- Advanced Learning Algorithms by Andrew Ng

#### Multiclass classification

Softmax:

Logistic regression returns the probability of y 	= 1, therefore it also returns the probability of y = 0

Let's review an example

Softmax regression for 4 possible outcomes	

1) $$ z_1 = w_1 * x + b_1 $$

2) $$ z_2 = w_2 * x + b_2 $$

2) $$ z_3 = w_3 * x + b_3 $$

4) $$ z_4 = w_4 * x + b_4 $$

The formula to compute the probability of y in each outcome is computed by using:

1) $$ a_1 = {e^{z1}\over e^{z_1} + e^{z_2} + e^{z_3} + e^{z_4}}  = P(y = 1|x) $$

1) $$ a_2 = {e^{z2}\over e^{z_1} + e^{z_2} + e^{z_3} + e^{z_4}}= P(y = 2|x)$$

1) $$ a_3 = {e^{z3}\over e^{z_1} + e^{z_2} + e^{z_3} + e^{z_4}}= P(y = 3|x) $$

1) $$ a_4 = {e^{z4}\over e^{z_1} + e^{z_2} + e^{z_3} + e^{z_4}}= P(y = 4|x) $$

General formula based on these assumptions:

$$ z_j = w_j * x + b_j $$

$$ a_j = {e^{zj}\over \sum^N_{k=1} e^{zk}} = P(y	=j|x) $$

***Cost function***
![softmax_cost](https://user-images.githubusercontent.com/46135649/184257494-23b4fd74-932f-44fb-90dd-afd7dc0867c6.png)

**Neural Network with softmax output**

The hidden layers are calculated following the same rules of the previous examples. 

The output layer is calculated using ***softmax*** to get the probabilities of each one of the classes. 

The Python code for a multi-classification problem is:

1) Specify the model:
```python
# 1- specify the model
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
model = Sequential([
	Dense(units=25, activation='relu')
	Dense(units=15, activation='relu')
	Dense(units=10, activation='softmax')
	])
```
2) Specify the loss and cost 
```python 
from tensorflow.keras.losses import SparseCategoricalCrossentropy
model.compile(loss=SparseCategoricalCrossentropy())
```
3) Train on data to minimize J(w,b)

```python
model.fit(X, Y, epochs=100)
```

This is not the most efficient way of implementing the code since some numerical errors might occur, the most efficient way is:
```python
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
model = Sequential([
	Dense(units=25, activation='relu')
	Dense(units=15, activation='relu')
	Dense(units=10, activation='linear')
	])
from tensorflow.keras.losses import SparseCategoricalCrossentropy
model.compile(loss=SparseCategoricalCrossentropy(from_logits=True))
model.fit(X, Y, epochs=100)
logits=model(X)
f_x =tf.nn.softmax(logits)
```

## Day 33

### 1- Advanced Learning Algorithms by Andrew Ng

Advance Optimization

***Adam algorithm***

Allow to optimize the convergent rate of gradient descent by increasing or decreasing alpha automatically. 

Adam: **Ada**ptive **M**ovement estimation

Python implementation of ADAM

```python
# Model
model = Sequential([
		tf.keras.layers.Dense(units=25,activation="sigmoid")
		tf.keras.layers.Dense(units=10,activation='sigmoid')
		tf.keras.layers.Dense(units=10,activation='linear')
# compile
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
	loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
#fit
model.fit(X,Y, epochs=100)
```

## Day 34

### 1- Advanced Learning Algorithms by Andrew Ng

Additional layer types

***Dense Layer***

Each neuron output is a function of all the activation outputs of the previous layer,

***Convolutional Layer***

Each neuron only looks at part of the previous layer's inputs.

- Faster computation
- Need less training data (less prone to overfitting)

The way it works is by looking at small windows, if we have a time series each neuron will look only to a small period divided by the amount of neuron on your layer. The output layer will be the classical layers, in which we decided based on the type of problem we have. 

## Day 35

### 1- Advanced Learning Algorithms by Andrew Ng

***Debuggin a learning algorithm***

- Get more training examples
- Try smaller sets of features
- Try getting additional features
- Try adding polynomial features
- Try decreasing $\lambda$
- Try increasing $\lambda$

Evaluating your model

***Regression***

Train/Test procedure for linear regression (with squared error cost)

Fit the parameters by minimizing cost function J(w,b)

**Test error**

$$ J_{test}(w,b) = {1\over 2m_{test}} \sum (f_{w,b}(x^i_{test}) - y^i_{test})^2 $$

**Train error**

$$ J_{train}(w,b) = {1\over 2m_{train}} \sum (f_{w,b}(x^i_{train}) - y^i_{train})^2 $$

***Classification***

**Test error**

$$ J_{test}(w,b) = -{1\over m_{test}} \sum [y^i_{test}log (f_{w,b}(x^i_{test}) ) + (1-y^i_{test}) log(1 - f_{wb}(x^i_{test}))] $$

**Train error**

$$ J_{train}(w,b) = -{1\over m_{train}} \sum [y^i_{train}log (f_{w,b}(x^i_{train}) ) + (1-y^i_{train}) log(1 - f_{wb}(x^i_{train}))] $$

## Day 36

### 1- Advanced Learning Algorithms by Andrew Ng

Model selection (choosing a model)

We can run different models one by one and pick the one that has the lowest cost function. However, this is an impractical approach. 

Alternative, we could divide the data into 
Training/ cross validation/ test set

From all this set, we calculate the loss function of each one of them.

Cross validation error/ dev error/ validation error

This can also be used to select the NN architecture -- all decisions needs to made based on the results on your training and cross validation set.

## Day 37

### 1- Advanced Learning Algorithms by Andrew Ng

Diagnosis bias and variance

**High bias (underfit)**

- Jtrain is high
- Jcv is high
- (Jtrain ~ Jcv)

**High variance (overfit)**
- Jtrain is low
- Jcv is high
- (Jtrain << Jcv)

**Just right**
- Jtrain is low
- Jcv is low

**High bias and High variance (NN applications)**
- Jtrain high
- Jcv much higher
- (Jtrain << Jcv)

There are some cases when you could also have high variance and high bias, mostly for neural network applications. 

## Day 38

### 1- Advanced Learning Algorithms by Andrew Ng

**Linear regresson with Regularization**

$\lambda$ is the parameter for regularization in the following equation that represents the lost function

$$ J(w,b) = {1\over 2m} \sum (f_{wb}(x^i) - y^{i})^2 + {\lambda\over 2m} \sum w_j^2 $$

Let's evaluate two possible scenarios:

1- We set lambda to a large number

If lambda is set too large, it would cause a huge penalty on the parameter of the regression, causing high bias and under fit the model.


2- We set lambda to a small number

There is no regularization term, so you will end up with high weights of the parameters in the prediction, causing high variance and overfit the data.

3- Intermediate values of lambda

Hopefully this will fit the data well

**Choosing the regularization parameter**

Cross-validation allow trying different set of values for lambda

**Establishing a baseline level of performance**

Having a benchmark to decide what went well and what went wrong is a critical aspect of ML model since it allow comparing the error rate.

What's the level of error you can reasonably hope to get to?

- Human level performance
- Competing algorithms performance
- Guess based on experience

Let's evaluate this example:

Baseline performance: 10.6%
Training error: 10.8%
Cross validation error: 14.8%

The training error is large as well as the cross validation error, so at first we might conclude that we have a bias error, or we are underfitting the model.

However, by setting a benchmark which is human error we can see that training error is actually good, so in fact the training error is low, and the CV error is large which is a case of high variance. 

Second example:
Baseline performance: 10.6%
Training error: 15%
Cross validation error: 15.5%

The training error is large but is close to CV error, indicating that there is an underfit problem (high bias)

Third example
Baseline performance: 10.6%
Training error: 15%
Cross validation error: 19.7%

Training error is large and CV is much larger than training error, this suggests a high variance and high bias problem. 

## Day 39

### 1- Advanced Learning Algorithms by Andrew Ng

Learning curves:

**High Bias (underfit)**

- Jtrain is high
- Jcv is high
- (Jtrain ~ Jcv)

If a learning algorithm suffers from high bias, getting more training data will not (by itself) help much. That's why it is crucial to identify the cause of the problem, so we don't spent resources on collecting more data when that won't solve our problem.

![image](https://user-images.githubusercontent.com/46135649/185795570-62650a0d-8545-4851-8717-a44563efe94f.png)

**High variance (overfit)**
- Jtrain is low
- Jcv is high
- (Jtrain << Jcv)

If a learning algorithm suffers from high variance, getting more training data is likely to help.

![image](https://user-images.githubusercontent.com/46135649/185795703-56246615-6dd6-4a08-a1bc-5202d9e512fd.png)

Debuggin a learning algorithm:

1. Get more training examples (Overfitting) → High variance
2. Try smaller sets of features (Overfitting) → High variance
3. Try getting additional features (Underfitting) → High Bias
4. Try additional polynomial features (Underfitting) -> High Bias
5. Try decreasing $\lambda$ (Underfitting) -> High Bias
6. Try increasing $\lambda$ (Overfitting) -> High variance

Bias variance tradeoff

Simple model -> High bias (underfit)
Complex model -> High variance (Overfit)

![image](https://user-images.githubusercontent.com/46135649/185932069-9acfa2cb-0f47-4441-bbc8-04c9c25b0e7a.png)

As general rule a large NN will usually do as well or better than a smaller one so long as regularization is chosen appropriately.

**Neural network regularization**

**Unregularized MINIST model**
```python
layer_1 = Dense(units=25, activation="relu")
layer_2 = Dense(units=15, activation="relu")
layer_3 = Dense(units=1, activation="sigmoid")
model = Sequential([layer_1, layer_2, layer_3])
```
**Regularized MINIST model**
```python
layer_1 = Dense(units=25, activation="relu", kernel_regularizer=L2(0.01))
layer_2 = Dense(units=15, activation="relu",kernel_regularizer=L2(0.01))
layer_3 = Dense(units=1, activation="sigmoid", kernel_regularizer=L2(0.01))
model = Sequential([layer_1, layer_2, layer_3])
```

**Iterative loop of ML development**

Choose architecture (model, data, etc.)

Train model

Diagnostics (bias, variance and error analysis)

**Error analysis**

Why is there a miss classification of the data by manually looking where are the discrepancies. 


## Day 40

### Adding data

**Augmentation**: modifying an existing training example to create a new training example.

If you find out your model is doing poorly on an especific label you might want to collect more training examples of your data but specific to that particular label. 

Examples on how to do it: Rotation of pictures, change the size of the pictures, adding distorition to pictures.

Speech recognition example:
add noise background of different types or bad cellphone connection

How to add noise data that makes sense

**note**: for market profile different markets?

Engineering  the data used by your system
Conventional model-centric approach: work on model
Data-centric approach: work on data

### Transfer learning

Is the process of using already trained models in a new data with less training examples.

Option 1: only train **output layers** parameters 
Option 2: train all parameters

Why does transfer learning work?
Low level layers learn to identify different type of parameters such as edges or corner, that information can be use on similar dataset, however we should use a model train on similar data to make it work.

Steps for transfer learning:
1. Download neural network parameters pretrained on a large dataset with same input type(e.g., images, audio, text) as your application (or train your own)
2. Further train (fine tune) the network on your own data.

### Full cycle of a machine learning project

1. Scope project (define project)
2. Collect data (Define and collect data)
3. Train model (Training, error analysis & iterative improvemnt) - this might take you to collect data based on the error analysis
4. Deploy in production (Deploy, monitor and maintain system)

![image](https://user-images.githubusercontent.com/46135649/186548831-0b436987-150b-4cec-aeca-a88acd206bd1.png)

**Deployment**

The deployment phase of your ML model requires the inference server which is where the ML model will be store, and the application in which you will have your input to make predictions. We make an API call to the inference server and returns the prediction to the APP. 

Software engineering may be needed for:
- Ensure reliable and efficient predictions
- Scaling
- Logging
- System monitoring
- Model updates


### Fairness, bias and ethics

Bias: 

Adverse use cases

**Guidelines:**

- Get a diverse team to brainstorm things that might go wrong, with emphasis on possible harm to vulnerable groups. 
-  Carry out literature search on standards/guidelines for your industry.
- Audit systems against possible harm prior to deployment.
- Develop mitigation plan (if applicable), and after deployment, monitor for possible harm. 

### Skewed datasets

Precision/recall

y = 1 in presence of rate class we want to detect

Let's say we train a learning algorithm achieving 99% correct diagnoses, however only 0.5% of the patients have the disease. 

If we set all predictions to y=0 we will achieve 99.5% accuracy/0.5% error

Confusion matrix helps to solve this issue on the Cross Validation set

**Actual class**
|1 |0 |
|--|--|
| True positive (15) | False Positive (5) 
| False negative (10) | True negative (70)

**Precision**: Of all patients where we predicted y = 1, what fraction actually have the rare disease?

$$ Positive_{true}\over Positive_{true} + Positive_{false} $$

$$ {15\over 15 + 5} = 0.75 $$

**Recall**: of all patients that actually have the rare disease, what fraction did we correctly detect as having it?

$$ Positive_{True}\over Positive_{True} + Negative_{False} $$

$$ {15\over 15+10} = 0.6 $$

## Trading off precision and recall

Precision = true positives/total predicted positive

Recall = true positives/total actual positive

Let's say we use logistic regression, for that we need to set a threshold:

Predict 1 if: $f_{w,b}(x) >= 0.5$

Predict 0 if: $f_{w,b}(x) < 0.5$

If we want to increase our confidence in the prediction of y = 1 (rare disease) we can increase our threshold.

Predict 1 if: $f_{w,b}(x) >= 0.9$

Predict 0 if: $f_{w,b}(x) < 0.9$

*This threshold will give higher precision but lower recall.*

Suppose we want to avoid missing too many causes of rare disease (when in doubt predict y = 1)

Predict 1 if: $f_{w,b}(x) >= 0.3$

Predict 0 if: $f_{w,b}(x) < 0.3$

*This threshold will give lower precision but higher recall*

Usually can be helpful to plot the curve between precision and recall and find a balance threshold that allow to pick the best threshold for our problems. 

**F1 Score**

F1 score is the metric that allow to automatize the process to pick the best trade-off opportunities. The equation for the F1 Score is the harmonic mean of P and R which enphasis in the smallest value. 

|  | Precision -P |Recall -R |F1 Score |
|--|--|--|--|
|Algorithm 1  | 0.5 |0.4 |0.444 |
|Algorithm 2  | 0.7 |0.1 | 0.175|
|Algorithm 3  |  0.02|1.0 | 0.0392|

$$ F_1 Score = {1\over {1\over 2} ({1\over P} + {1\over R} )} $$

$$ F_1 Score = 2 {PR\over P+R} $$


### Try many regularization values to find the optimal solution

```python
tf.random.set_seed(1234)
lambdas = [0.0, 0.001, 0.01, 0.05, 0.1, 0.2, 0.3]
models=[None] * len(lambdas)
for i in range(len(lambdas)):
    lambda_ = lambdas[i]
    models[i] =  Sequential(
        [
            Dense(120, activation = 'relu', kernel_regularizer=tf.keras.regularizers.l2(lambda_)),
            Dense(40, activation = 'relu', kernel_regularizer=tf.keras.regularizers.l2(lambda_)),
            Dense(classes, activation = 'linear')
        ]
    )
    models[i].compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(0.01),
    )

    models[i].fit(
        X_train,y_train,
        epochs=1000
    )
    print(f"Finished lambda = {lambda_}")
```

## Day 41

**Decision tree model**

Root node

Decision tree

Tree node

**Decision tree learning**

Decision 1: How to choose what feature to split at each node?

Maximize purity (or minimize impurity)

The feature that we want to choose in higher level nodes for decision tree should be the one that maximize the purity, which mean the one that get most of the right answers

Decision 2: When do you split?

- when a node is 100% one class

- when splitting a node will result in the tree exceeding a **maximun depth**

Maximun depth means the number of hops that it takes to get from the root node. 
- When improvements in purity score are below a threshold
- when number of examples in a node is below a threshold

**Measure of impurity**

To measure the impurity we use the entropy function. The entropy function returns a value for each probability of correct classification. 

$$ H(p_1) = -p_1 log_2(p_1) - (1-p1)log_2(1-p1) $$

Note: By convention "0 log(0)" = 0


**Choosing a split information gain**

We compare each feature and the degree of impurity based on the previous formula. For this we take a weigthed average the number of correct classification divided by the sample size.

The way we decided which one will be the root node we need to choose based on the reduction of entropy if we haven't split at all (Information gain). One parameter to stop the training is if the information gain is too small stop the training to prevent overfitting. 

![image](https://user-images.githubusercontent.com/46135649/187892136-dd901914-8b75-4796-8745-265bc5db3712.png)

Now let's answer: How a Decision Tree learn?

- Start with all examples at the root node
- Calculate the information gain for all possible features, and pick the one with the highest information gain. 
- Split dataset according to selected feature, and create left and right branches of the tree
- Keep repeating spliting process until stopping criteria is met:
	- When a node is 100% one class
	- When splitting a node will result in the tree exceeding a maximun depth
	- Information gain from additional splits is less than threshold
	- When number of examples in a node is below a threshold

Decision tree are recursive algorithm in which each branch is another decision tree. -code that cause it self-

The depth of a decision tree is like training a higher degree polynomial which make a complex model but increase the chance of overfitting. 

## Day 42

### One hot encoding of categorical features

Instead of having a single feature with multiple outcome we can have several feature 

### Continuos valued features

Splitting on a continuos variable

We calculate the information gain by setting a threshold, this threshold will be evaluated based on how well classifies our prediction. 

Let's see an example

We have 5 cats and 5 not cat

We have a continuos feature (weight lbs)

If we set a threshold at 8 allowing to correctly classify 2 cats out of 10 examples

The entropy at the root node is 0.5 

**Left wind**

Proportion of examples in the category $2\over 10$ regardless if there are correct or not

Entropy of $2\over2$ -> this mean the proportion of y(1) examples in the category.

**Right wind**

Proportion of examples in the category $8\over 10$ 

Entropy $3\over8$ y(1) examples in the category by total examples.


$$ H(0.5) - ({2\over 10} H ({2\over 2}) + {8\over 10}H ({3\over 8})) $$


Threshold: 9

$$ H(0.5) - ({4\over 10} H ({4\over 4}) + {6\over 10}H ({1\over 6})) $$


