import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

np.random.seed(6740)
data = np.random.randn(100, 2) #@ np.array([[1, 0.8], [0.8, 1]])

pca_clean = PCA(n_components=2)
pca_clean.fit(data)
outlier = np.array([[4, -5]])
data_outlier = np.vstack([data, outlier])

pca_outlier = PCA(n_components=2)
pca_outlier.fit(data_outlier)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(data[:, 0], data[:, 1], label='Clean data')
pc1_clean_direction = pca_clean.components_[0]
plt.quiver(data.mean(), data.mean(), pc1_clean_direction[0], pc1_clean_direction[1],
           angles='xy', scale_units='xy', scale=1, color='red',
           label='PC1 Direction')
plt.title('PCA on Clean Data')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.scatter(data_outlier[:, 0], data_outlier[:, 1], alpha=0.7, label='Data')
plt.scatter(outlier[0, 0], outlier[0, 1], color='orange', s=100,
            label='Outlier')
pc1_outlier_direction = pca_outlier.components_[0]
plt.quiver(data_outlier.mean(), data_outlier.mean(), pc1_outlier_direction[0], pc1_outlier_direction[1],
           angles='xy', scale_units='xy', scale=1, color='red',
           label='PC1 Direction')
plt.title('PCA on Data with Outlier')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
