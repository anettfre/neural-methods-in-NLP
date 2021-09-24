import matplotlib.pyplot as plt
import math

plt.rcParams['font.family'] = "Arial"

time = [7, 9, 11, 12, 14]
accuracy = [0.491, 0.513, 0.500, 0.479, 0.474]
precision = [0.416, 0.443, 0.420, 0.422, 0.404]
recall = [ 0.401, 0.433, 0.420, 0.403, 0.399]
f1 = [0.401, 0.430, 0.413, 0.400, 0.389]

a = plt.plot(time, accuracy, color='blue', label='Accuracy')
b = plt.plot(time, precision, color='black', label='Precision')
c = plt.plot(time, recall, color='red', label='Recall')
d = plt.plot(time, f1, color='green', label='F1 score')

plt.legend()

plt.xlabel("Training time", fontsize = 14)
plt.ylabel("Evaluation metrics", fontsize = 14)

plt.show()

