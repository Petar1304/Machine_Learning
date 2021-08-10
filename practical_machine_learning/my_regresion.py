import numpy as np
import random

x = np.array([1, 2, 3, 4, 5, 6], dtype=np.float64)
y = np.array([5, 4, 6, 5 ,6, 7], dtype=np.float64)


def best_fit_slope_and_intersect(x, y):
	m = ( np.mean(x) * np.mean(y) - np.mean(x*y) ) / (np.mean(x)**2 - np.mean(x**2))
	b = np.mean(y) - m * np.mean(x)
	return m, b


def squared_error(y, y_line):
        return sum((y_line - y)**2)

def coefficient_of_determination(y, y_line):
        y_mean_line = np.mean(y)
        squared_error_reqr = squared_error(y, y_line)
        squarred_error_y_mean = squared_error(y, y_mean_line)
        return 1 - (squared_error_reqr / squarred_error_y_mean)

        
m, b = best_fit_slope_and_intersect(x, y)
regression_line = [(m*i) + b for i in x]

print(f'y = {m}x + {b}')


r_squared = coefficient_of_determination(y, regression_line)

print(r_squared)
