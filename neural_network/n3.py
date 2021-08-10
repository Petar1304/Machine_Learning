import numpy as np
import nnfs

nnfs.init()

layer_outputs = [[4.8, 1.21, 2.385],
				 [8.9, 1.81, 0.2], 
				 [1.41, 1.051, 0.026]]


# E = np.e 
# exp_values = []
# for output in layer_outputs:
# 	exp_values.append(E**output)
# norm_base = sum(exp_values)
# norm_values = []
# for value in exp_values:
# 	norm_values.append(value / norm_base)
# norm_values = [val / sum(exp_values) for val in exp_values]

# with numpy
exp_values = np.exp(layer_outputs)
norm_values = exp_values / np.sum(exp_values, axis=1, keepdims=True)

print(norm_values)

# print(np.sum(norm_values, axis=1, keepdims=True))