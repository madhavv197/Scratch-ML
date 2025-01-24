![Banner](assets/banner(2).png)


# About
Scratch-ML is a personal project I made, because I was using ML libraries blindly without really understanding what was happening. I decided to rebuild everything from scratch using just NumPy. Each implementation here is me figuring out how these algorithms actually work, rather than just calling model.fit() and hoping for the best.

# Neural Network Architecture

A neural network consists of a series of layers, the input layer, hidden layers, and output layer, that contain a series of nodes (neurons), that perform operation:

$$
z = wx+b
$$

Each neuron is connected to each input $x_i$, with each connection having some weight, we can call $w_i$. Each neuron also has some bias $b$, that it adds on to the $weight \cdot input$. So for each layer of neurons, as a matrix operation, we have that its output $z$ will be equal to:

$$
Z = WX + B
$$


In essence, neural networks are essentially a series of linear regressions, with one key difference, the output of each node, $z$, is made non-linear into what we call an activation, $a$, using some activation function after the linear operation, $f(z)$. (Make sure you know the difference between z and a, this becomes important for later.)

These activations, $a$ are then passed on to the next layer as input, for it to calculate it's $z$, which will then be used as input, so on and so on. Eventually, these activations will reach the final layer, in which case, we may decide to use a output function instead of an activation function. From this, we have calculated our prediction.

This is known as our "forward pass". 

From our prediction we want to know how wrong we were from the true value. We do this using a loss function, which we will call $C(a)$ or just $C$. From our loss, we want to update our weights in such a way that we try to minimize this loss, to make our prediction more accurate. We now calculate the gradient $\frac{\partial C}{\partial w}$, to know how much to "step" our weights and biases to decrease the loss, we do this by the following:

$$
w = w - \eta \frac{\partial C}{\partial w}
$$


$$
b = b - \eta \frac{\partial C}{\partial b}
$$

This is known as the "backward pass" and "gradient descent".

This process is repeated iteratively, in order to find a model as accurate as possible. Of course it can't just lower it's loss to 0 from an infinite number of iterations. Firstly, the model may get stuck in a local minimum during gradient descent, or the (bigger) problem, it ay start overfitting, where it just memorises different patterns in the training data, and it is unable to predict accurately on new, unseen data.

Below is an explanation of an example coded up in the github, using the mnist dataset, which is a series of handwritten numbers from 0-9.

# Forward Pass

In this code, we flatten our mnist dataset into a (1x784) matrix that we then pass as our first input.

As mentioned earlier, this step involves computing all the linear regressions, and passing them through the activation function for each neuron in each layer in the network. First we apply the regressions:

$$
Z = WZ + B
$$


Lets call $Z$ the "pre-activations" of a neuron.

$Z$ is then passed through the activation function $f(z)$. In this example, lets use the ReLU function as our activation function, which is: 

$$
f(z) = ReLU(z) = max(0,z)
$$

 For our loss we use the cross entropy loss for multi-class classification this is given by (for a single sample):

$$
L = - \sum_{i=1}^C y_i \log(\hat{y}_i)
$$

In order for this to work, the final output layer must give a (10x1) vector, to represent 10 classes.

# Backward Pass

In order to find the gradients of the cost w.r.t to the weights, $\frac{\partial C}{\partial w}$, we need to use the chain rule. Our formulation for this will be different for the output layer compared to the other layers, as we use a softmax instead of a ReLU, and so we will start with the output layer. We have that a general formulation for the gradient of the cost is given by:

$$
\frac{\partial C}{\partial w} = \frac{\partial C}{\partial a} \frac{\partial a}{\partial z} \frac{\partial z}{\partial w}
$$

Starting with the easy part, we can find that $\frac{\partial z}{\partial w}$ is given by:

$$
\frac{\partial z^l}{\partial w^l} = \frac{\partial (wa^{l-1}+b)}{\partial w^l} = a^{l-1}
$$

Which is the activations from the previous layer. 

Now for the output layer we can say that $\frac{\partial C}{\partial a}$:

$$
\frac{\partial C}{\partial a} = \frac{\partial (- \sum_{i=1}^C y_i \log(\hat{y}_i))}{\partial a}
$$

For some prediction, there will be only 1 correct label with value 1, and everything else will be 0, hence $y_i$ will be 0 for all $i$ except for some $n$ which is the label. Thus our sum will simplify into:

$$
\frac{\partial C}{\partial a} = \frac{\partial (- log(\hat{y_n}))}{\partial a} = - \frac{1}{\hat{y_n}}
$$

$$
a = \frac{e^{z_i}}{\sum_{j=1}^C e^{z_j}}
$$
Again for some $i$ and some $n$,

$$
\frac{\partial a}{\partial z} = \frac{e^{z_i} (C+e^{z_i}) - e^{2z_i}}{(C+ e^{z_j})^2} = a_i (1-a_i)
$$

$$
\frac{\partial C}{\partial a} \frac{\partial a}{\partial z} = a_i-1
$$

Finally:

$$
\frac{\partial C}{\partial w} = \frac{\partial C}{\partial a} \frac{\partial a}{\partial z} \frac{\partial z}{\partial w} = (a_i-1)(a_i^{l-1})
$$




# Network Training
