# Building a neural network from scratch

### Purpose and Objective:
The goal is to build a simple multilayer perceptron (neural network) model from first principles and use it to effectively classify/identify handwritten images of single number digits.

**Why?** To showcase an understanding of the mathematics & fundamental algorithms (e.g. forward prop, back prop, gradient descent) underpinning neural networks often overlooked when implementing neural nets utilising TensorFlow, Keras etc.

**The only library used:**
Only NumPy will be used to build our model. Matplotlib and time libraries are non-mandotory and only used for visualisation and model training time purposes.

```python
import numpy as np  # for linear algebra operations
import matplotlib.pyplot as plt  # for some plots
import time # to check how long it takes our final model to train
```

### Training and testing our model on a simple visualisation task:
We train our model on the MNIST dataset which is a large database of handwritten digits that is often used for training image processing models. The train our model to effectively learn to recognise the digits 0 through to 9.

**Our main conclusions in brief:**

1. We build a model with 94% accruacy across both training and test sets. This suggests good stability and showcases the ability of neural nets to learn effectively with limited computational power and model complexity.

2. It will be shown that the types of images the model struggles to identify are ambigious and with "poor handwritting". 

3. We end by discussing some simple ways to improve the model to better handle these more difficult cases.

### Acknowledgements

Data downloaded from Kaggle: https://www.kaggle.com/datasets/hojjatk/mnist-dataset

**Acknowledgements as per Kaggle website:**

- Yann LeCun, Courant Institute, NYU
- Corinna Cortes, Google Labs, New York
- Christopher J.C. Burges, Microsoft Research, Redmond
