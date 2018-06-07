import sys
import matplotlib.pyplot as plt
import numpy as np

true_filename = sys.argv[1]
predicted_filename = sys.argv[2]
search_example = int(sys.argv[3])

true_array = np.load(true_filename)
predicted_array = np.load(predicted_filename)

assert true_array.shape == predicted_array.shape

print(true_array.shape)
print(predicted_array.shape)

trues = []
predictions = []

for i in range(len(true_array)):
    trues.append(true_array[i, search_example])
    predictions.append(predicted_array[i, search_example])

trues = np.asarray(trues)
predictions = np.asarray(predictions)
indices = range(len(trues))
color = [index / len(indices) for index in indices]
plt.plot(trues[:, 0], trues[:, 1], c='black', marker="P")
plt.plot(predictions[:, 0], predictions[:, 1], c='red')
plt.show()
