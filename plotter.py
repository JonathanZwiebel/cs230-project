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

counts = 0
sum_squared_error = 0
zero_error = 0

for i in range(len(true_array)):
    sum_squared_error = sum_squared_error + (true_array[i, search_example, 0] - predicted_array[i, search_example, 0]) ** 2
    sum_squared_error = sum_squared_error + (true_array[i, search_example, 1] - predicted_array[i, search_example, 1]) ** 2
    zero_error = zero_error + (true_array[i, search_example, 0]) ** 2
    zero_error = zero_error + (true_array[i, search_example, 1]) ** 2
    counts = counts + 2
    if i % 10 == 0 or i == range(len(true_array)):
        trues.append(true_array[i, search_example])
    predictions.append(predicted_array[i, search_example])

print(sum_squared_error / counts)
print(zero_error / counts)

trues = np.asarray(trues)
predictions = np.asarray(predictions)
indices = range(len(trues))
color = [index / len(indices) for index in indices]
plt.plot(trues[:, 0], trues[:, 1], c='black', marker="P")
plt.plot(predictions[:, 0], predictions[:, 1], c='red')
plt.show()
