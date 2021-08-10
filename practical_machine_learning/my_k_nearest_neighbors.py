import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from collections import Counter
import warnings

style.use('fivethirtyeight')

dataset = {'k': [[1, 2], [2, 3], [3, 1]], 'r':[[6, 5], [7, 7], [8, 6]]}
new_features = [5, 7]

# for i in dataset:
	# for j in dataset[i]:
		# plt.scatter(j[0], j[1], s=100, color=i)
# plt.show()

def k_nearest_neighbors(data, predict, k=3):
	if len(data) >= k:
		warning.warn('K is set to a value less then total voting groups')
	distances = []
	for group in data:
		for features in data[group]:
			euclidean_distance = np.linalg.norm(np.array(features) - np.array(predict))
			distances.append([euclidean_distance, group])

	votes = [i[1] for i in sorted(distances)[:k]]
	vote_result = Counter(votes).most_common(1)[0][0]
	return vote_result

result = k_nearest_neighbors(dataset, new_features)
print(result)

