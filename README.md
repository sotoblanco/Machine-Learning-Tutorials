# Machine-Learning-Tutorials

## Advance Learning Algorithms Coursera by Andrew Ng

### Day 26:

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





