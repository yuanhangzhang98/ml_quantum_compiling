# Topological quantum compiling with reinforcement learning
An efficient machine learning algorithm to decompose an arbitrary single-qubit gate into a sequence of gates from a finite universal set. Reference: https://arxiv.org/abs/2004.04743

In this example code the universal gate set is chosen to be the braiding operations of the Fibonacci anyon model. 

## Usage
To train a model from scratch: 
```
python3 main.py
```

To test a pretrained model on randomly generated matrices:
```
python3 test.py
```

Sorry that I haven't provided any convenient tools for customizing the model yet. In order to decompose a particular quantum gate, or training a new model, you can just clone the source code and edit the corresponding parts. 
