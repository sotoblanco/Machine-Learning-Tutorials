# Machine-Learning-Tutorials

## Day 26:

### Advance Learning Algorithms Coursera by Andrew Ng

#### Neuronal Network Layer

Each NN layer takes an input and returns an output, this ouput is call the activation value and is calculated using this formula:

$a = g(w * a + b)$

where g is the sigmoid function also call activation function and the parameters are taking from the logistic regression formula:

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


### TensorFlow Developer Certificate by Zero to Mastery

#### Steps to improve a model

The common ways to improve a model are:

- Adding layers
- Increase the number of hidden units
- Change the activation functions
- Change the optimization function
- Change the learning rate (This is perhaps the most important parameter)
- Fitting on more data
- Fitting for longer

We need to go each of this one by one and adjusting as we see the results

Sometimes the model with more layers, small learning rate and high number of epochs is not the best one, so that's why we need to keep evaluating with each metric one by one and see what works. 

![Improve NN](https://user-images.githubusercontent.com/46135649/182184699-d1f67197-190d-44ff-876a-6b50f40b1d2c.png)


### Machine Learning with Python by FreeCodeCamp

#### NLP with RNNs

Natural Language Processing

We need to change each word to a numerical value, the dataset use is a review of movies from TensorFlow library. All reviews must have the same length so the data needs to be standardize. The way it works is by:

- if the review is greater than 250 words then trim off the extra words
- if the review is less than 250 words add the necessary amount of 0's to make it equal to 250.

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








