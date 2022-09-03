## Day 30

### 2- TensorFlow Developer Certificate by Zero to Mastery

TensorFlow allows visualizing the model and customize the names for a better explanation on how we build the model. 

Let's use an example without data by using just the input shape of the model:

```python
import tensorflow as tf
# build the model
model = tf.keras.Sequential([
tf.keras.layers.Dense(10, input_shape=[1], name="input_layer"),
tf.keras.layers.Dense(1, name="output_layer")
], name="model_1")

# compile the model
model.compile(loss=tf.keras.losses.mae,
optimizer=tf.keras.optimizers.SGD(),
metrics=["mae"])

model.summary()
```
This creates a model summary with layers, the output shape of each layer, and the parameters. The total parameters divided by the trainable parameters and non-trainable parameters.

**Plot the model**
```python
from tensorflow.keras.utils import plot_model
plot_model(model=model, show_shapes=True)
 ```

## Day 31

### 2- TensorFlow Developer Certificate by Zero To Mastery

#### Visualize models performance

```python
def plot_predictions(train_data=X_train,
					 train_labels=y_train,
					 test_data = X_test,
					 test_labels=y_test,
					 predictions=y_pred):

	"""
	Plots training data, test data
	"""
	plt.figure(figsize=(10,7))
	# plot training data in blue
	plt.scatter(train_data, train_labels, c="b", label="Training data")
	# plot testing data in green
	plt.scatter(test_data, test_labels, c= "g", label="Testing data")
	# plot model predictions in red
	plt.scatter(test_data, predictions, c= "r", label="Predictions")
	# show legend
	plt.legend();
```

## Day 32
### 2- TensorFlow Developer Certificate by Zero To Mastery

***Evaluation metrics***

There are different evaluation metrics depending on the problem you're working on. 

- MAE - mean absolute error: on average how wrong is each of my model's predictions
- MSE - mean square error, "Square the average errors"

Make sure y_test and y_pred has the same shape of tensor, if not test ``tf.squeeze`` to reduce the tensor using just the shape needed.


| Metric name | Metric formula | TensorFlow code | When to use |
|--|--|--|--|
| Mean absolute error (MAE) | $$ MAE = {\sum ^n_{i=1} \|y_i - Y_i\| \over n} $$  |``tf.keras.losses.MAE()`` or ``tf.metrics.mean_absolute_error() ``  | As a great starter metric for any regression problem |
| Mean square error (MSE) | $$ MSE =  {1\over n} {\sum ^n_{i=1} (y_i - Y_i)^2} $$  |``tf.keras.losses.MSE()`` or ``tf.metrics.mean_square_error() ``  | When larger errors are more significant than smaller errors. |
| Huber  | |``tf.keras.losses.Huber()`` | Combination of MSE and MAE. Less sensitive to outliers than MSE.|

## Day 33

### 2- TensorFlow Developer Certificate by Zero To Mastery

Experiments to improve our model

1. Get more data
2. Make your model larger
3. Train for longer

Example of model experiment:

1. Model 1: 1 layer 100 epochs
2. Model 2: 2 layers 100 epochs
3. Model 3: 2 layers 500 epochs

```python
model_1 = tf.keras.Sequential([
	tf.keras.layers.Dense(1)
	])
	
model_1.compile(loss=tf.keras.losses.mae,
	optimizer=tf.keras.optimizers.SGD(),
	metrics=["mae"])
	
model_1fit(X_train, y_train, epochs=100)

y_preds_1 = model_1.predict(X_test)
# previous created function to plot predictions
plot_predictions(predictions=y_preds_1) 
#calculate MAE and MSE
```

## Day 37
### 2- TensorFlow Developer Certificate by Zero To Mastery

```python
#set the random seed
tf.random.set_seed(42)

# 1. Create a model
model_3 = tf.keras.Sequential([
	tf.keras.layers.Dense(10),
	tf.keras.layers.Dense(1)
	])
	
# 2. Compile the model
model_3.compile(loss=tf.keras.losses.mae,
				optimizer=tf.keras.optimizers.SGD(),
				metrics=["mae"])
# fit the model
model_3.fit(X_train, y_train, epochs=500)
```
## Day 38
### 2- TensorFlow Developer Certificate by Zero To Mastery

Compare your models with a Pandas dataframe

```python
model_results = [["model_1", mae_1.numpy(), mse_1.numpy()],
				["mode_2", mae_2.numpy(), mse_1.numpy()],
				["model_3", mae_3.numpy(), mse_3.numpy()]]

all_results = pd.DataFrame(model_results, columns=["model", "mae", "mse"])
```
As we increase the number of experiments, it becomes a tedious task to do, there are some tools design to help us.

* TensorBoard - a component of TensorFlow library to help track modeling experiments.
* Weights & Biases - a tool for tracking all the kinds of ML experiments (plugs straight into tensorboard)

***Save & export the model***: 

This allows to save your model and use it outside your Google Colab

Two main formats we can save our model:

1. The SavedModel format
2. The HDF5 format

Saved format
```python
model_2.save("best_model_savedmodel_format")
```
HDF5 format
```python
model_2.save("best_model_HDF5_format.h5")
```

## Day 39

Loading a save model

```python
# Load in the SavedModel format model
loaded_SavedModel_format = tf.keras.models.load_model("best_model_SavedModel_format")
```

## Day 40

Let's use a real dataset from kaggle [Medical Cost Personal dataset](https://www.kaggle.com/datasets/mirichoi0218/insurance) 

This is a Regression problem:

Import libraries
```python
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
```
Read the dataset
```python
insurance = pd.read_csv("https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv")
```
We need to predict the charges of a person based on several features. The overall shape of this dataset is 1338 rows and 7 columns. 

First we need to process the data, and that start with encode categorical features as dummy features
```python
# Numerical encoding
insurance_one_hot = pd.get_dummies(insurance)
insurance_one_hot.head()
```
Set X and y label

```python
# Create X & Y (features and labels)
X = insurance_one_hot.drop("charges", axis=1)
y = insurance_one_hot["charges"]
```
Split your dataset in training and testing: This can be improve using validation set as well.

```python
# Create training and test set using sklearn which will return a random split of training and testing set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
len(X),  len(X_train),  len(X_test)
```
Build and train your model - Also prepare the other experiments that you would like to run
```python
# Build a neural network
tf.random.set_seed(42)
# 1. Create a model
insurance_model = tf.keras.Sequential([
tf.keras.layers.Dense(10),
tf.keras.layers.Dense(1)
])
# 2. Compile the model
insurance_model.compile(loss=tf.keras.losses.mae,
optimizer=tf.keras.optimizers.SGD(),
metrics=["mae"])
# 3. Train the model
insurance_model.fit(X_train, y_train, epochs=100)
```
Test the results of your model
```python
# check the results of the insurance model on the test data
history = insurance_model.evaluate(X_test, y_test)
```
Plot the training curve
```python
# plot the training curve
pd.DataFrame(history.history).plot()
plt.ylabel("loss")
plt.xlabel("epochs")
```

## Day 41

### Steps in modeling with TensorFlow

1. Get data ready (Turn into tensors)
2. Build or pick a pretrained model
3. Fit the model to the data and make prediction
4. Evaluate the model
5. Improve through experimentation
6. Save and reload your trained model

### Preprocessing data with feature scaling

|Scaling type|What it does  |Scikit-Learn Function |When to use |
|--|--|--|--|
|Scale(also refered to as normalization  | Converts all values to between 0 and 1 while preserving the original distribution  |MinMaxScaler |Use as default scaler with neural networks | |
|Standardization  | Removes the mean and divides each value by the standard deviation |StandardScaler |Transform a feature to have close to normal distribution (caution: this reduces the effect of outliers) | |

Example on how to prepare your data before running a model

```python
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

# Create a column transformer
ct = make_column_transformer(
	(MinMaxScaler(),  ["age",  "bmi",  "children"]),  # turn all values in these columns between 0 and 1
	(OneHotEncoder(handle_unknown="ignore"),  ["sex",  "smoker",  "region"]))

# create X & y
X = insurance.drop("charges", axis=1)
y = insurance["charges"]

# build our train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the column transformer to our training data
ct.fit(X_train)

# Transform training and test data with normalization (MinMaxScaler) and OneHotEncoder
X_train_normal= ct.transform(X_train)
X_test_normal = ct.transform(X_test)
```

