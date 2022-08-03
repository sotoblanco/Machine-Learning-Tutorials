# Machine-Learning-Tutorials

## Day 26:

### 1- Advance Learning Algorithms Coursera by Andrew Ng

#### Neuronal Network Layer

Each NN layer takes an input and returns an output. This output is called the activation value and is calculated using this formula:

$a = g(w * a + b)$

Where g is the sigmoid function, also called activation function, and the parameters are taken from the logistic regression formula:

$g(z) = 1/over 1 + e^{-(z)}$

This returns an activation value that forms a vector if we have more than one neuron:

$a = [0.7, 0.3, 0.2]$

This vector is taken as input for the next layer

The activation value recieves a subscript that denotes the layer in which was obained, so the second layer will have this formula:

$a = g(w * a^{[1]} + b)$

If we only have one neuron the return will be a scalar value 

a = 0.82

The last step which is optional is a classification process for which we need to set a threshold:

is a >= 0.5

yes: y = 1

No: y = 0

The general formula for the activation value is:

$a^{[l]} = g(w^{[l]} * a^{[l-1]} + b^{[l]})$

The below graph shows the general architecture of a complex NN that use the above formula on each of the layers and neurons

![NN_complex](https://user-images.githubusercontent.com/46135649/182147605-17f0f0b9-5c70-4dbb-91a7-e73da330be19.png)


### 2- TensorFlow Developer Certificate by Zero to Mastery

#### Steps to improve a model

The common ways to improve a model are:

- Adding layers
- Increase the number of hidden units
- Change the activation functions
- Change the optimization function
- Change the learning rate (This is perhaps the most important parameter)
- Fitting on more data
- Fitting for longer

We must go through each of these individually and adjust as we see the results.

Sometimes the model with more layers, a small learning rate, and a high number of epochs is not the best one, so that's why we need to keep evaluating each metric one by one and see what works. 

![Improve NN](https://user-images.githubusercontent.com/46135649/182184699-d1f67197-190d-44ff-876a-6b50f40b1d2c.png)


### 3- Machine Learning with Python by FreeCodeCamp

#### NLP with RNNs

Natural Language Processing

We need to change each word to a numerical value. The dataset used is a review of movies from the TensorFlow library. All reviews must have the same length, so the data needs to be standardized. The way it works is by:

- if the review is greater than 250 words, then trim off the extra words
- if the review is less than 250 words, add the necessary amount of 0's to equal 250.

Which can be achive by using 
```
from keras.preprocessing import sequence
train_data = sequence.pad_sequences(train_data, 250)
```
This is the overall look of the model for NLP

The Embedding layer find a more meaningfull representation of our current data into vectors of 32 dimensions
We Long Short Term Memomry for our neurons (so they can learn from the previous output)
The activation function is the sigmoid function which allow to have a value between 0 and 1

```
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(VOCAB_SIZE, 32),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(1, activation="sigmoid")
])
```

We train the data by setting up the parameters of the loss function, the optimizer, and the metrics

```
model.compile(loss="binary_crossentropy",optimizer="rmsprop",metrics=['acc'])
```

We pass the training data, the epochs and choose the amount of data use for the validation data

```
history = model.fit(train_data, train_labels, epochs=10, validation_split=0.2)
```

### 4- Design Machine Learning Systems by Chip Huyen

#### Chapter 3. Data Engineering Fundamentals

![sqlvsnonsql](https://user-images.githubusercontent.com/46135649/182207369-f9e019e7-d8c3-4c18-9c5c-314ab068ef50.png)

The above table shows the difference between Strcuture data and unstructure data. This is a critical aspect of ML systems since is going to decide how are we going to store our data and provide to the different departments. 

Unstructured data allow flexibility. If things on the data change, we can easily customize with unstructured data since all our data doesn't need to follow the same schema. On the other hand, structure data follows the same schema, so if one thing changes, all the data needs to be customized to incorporate the change. Usually, structured data is used to store data that has been processed and sent to the teams for analysis, while non-structured data is raw data that needs some type of transformation. 


Types of workloads:
- Transactional Processing
- Analytical Processing


### 5- Crash Course Machine Learning by Google Developers 

#### Multi-Class Neural Nets

Often, we might need to classify things with one or more labels. This can be achieved with the  One vs. all approach to leverage binary classification, in which we create a binary classifier for each possible outcome. 

This approach is reasonable with a few classes but becomes increasingly inefficient as the number of classes rises. 

![one_vs_all](https://user-images.githubusercontent.com/46135649/182247594-f57b092f-d868-4bf8-88be-7428c855e2bb.png)

The second approach is **Softmax** in which we extend the idea of the logistic regression into a multiclass world.

**Softmax options**

-Full softmax: calculates the probability for every possible class
-Candidate sampling: means that softmax calculates a probability for all possible labels but only for a random sample of negative labels. This approach can improve efficiency with large number of classes. 

Softmax assumes that each example belong to exactly just one class, however if the problem has more than one class you may not use softmax and you most rely on multiple logistic regressions.

To access to the pictures in a dataset after each picture is code by a number you can use:

`` matplotlib.pyplot.imshow `` which allow to interpret numeric array as an image

```
# import modules
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# get the mnist dataset
(x_train, y_train),(x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Show image
plt.imshow(X_train[2917])
```

## Day 27

### 1- Advance Learning Algorithms Coursera by Andrew Ng

Forward propagation is a sequence of steps in Neural Networks to return an output. 

These are the step for build a NN:

- Select the number of hidden layers
- Select the neurons in each hidden layer

Behind scenes the calculation will be the activation value in each layer that will serve as input for the next layer, this type of sequential steps is called forward propagation. 

![foward_propagation](https://user-images.githubusercontent.com/46135649/182459488-20ed66a0-dc75-4d75-8bc9-b292f9f9b9d3.png)


### 2- TensorFlow Certificate by Zero to Mastery

Evaluating a TensorFlow model

For evaluating a TF model we need to visualize

- The data - what data are we working with? What does it look like?
- The model itself - what does our model look like?
- The training of a model - how does a model perform while it learns?
- The predictions of the model - how do the predictions of a model line up against the ground truth

### 3- Designing Machine Learning Systems by Chip Huyen

**Transactional databases** are designed to have low latency and high availability requirements. Transactional databases has 4 characteristics:

- Atomicity: All steps in a transaction are completed as a group
- Consistency: All transactions must follow predefined rules
- Isolation: To guarantee that two transactions happen at the same time
- Durability: To guarantee that once a transaction has been committed, it will remain committed even in the case of a system failure.

> An interesting paradigm in the last decade has been to decouple storage from processing (also known as compute), as adopted by many data vendors including Google’s BigQuery, Snowflake, IBM, and Teradata. In this paradigm, the data can be stored in the same place, with a processing layer on top that can be optimized for different types of queries.


### 4- Machine Learning with Python by FreeCodeCamp

NLP allows evaluating the sentiment of words in a sentence by converting them into integers. The output is a numerical value between 0 and 1 that denotes the probability of a positive sentence. We can add a threshold to our decisions to classify them into positive or negative.

### 5- Machine Learning Crash Course by Google Developers

**Embeddings** Allow you ow to transform your data into numerical values and represent them as numerical values, from which you can calculate how similar they are based on location in a multidimensional space. Embeddings make it easier to do ML on large inputs. 

One thing to remember is that the dimensions representing your data are hyperparameters that can be optimized. As a rule of thumb, you want to apply the fourth root of the possible values in our outcome. 

-Higher dimesional embeddings can represent accurately the representation between input values

-More dimensions increase the chance of overfitting and leads to slower training




