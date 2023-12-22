import matplotlib.pyplot as plt
import numpy as np

algorithms = ['SVM', 'Naive Bayes', 'Random Forest', 'KNN']
metrics = ['Accuracy', 'Precision', 'Recall', 'F-score']
data = np.array([
    [0.7619, 0.76, 0.76, 0.76],  # SVM
    [0.7143, 0.715, 0.71, 0.71],  # NB
    [0.6190, 0.62, 0.62, 0.61],  # RF
    [0.9048, 0.92, 0.90, 0.90]   # KNN
])

colors = ['blue', 'red', 'green', 'orange', 'purple']
plt.figure(figsize=(10, 6))

for i in range(len(algorithms)):
    plt.plot(metrics, data[i], label=algorithms[i], marker='o', color=colors[i])

plt.title('Performance Metrics for Different Algorithms')
plt.xlabel('Metrics')
plt.ylabel('Score')
plt.legend()
plt.grid(True)
plt.show()
