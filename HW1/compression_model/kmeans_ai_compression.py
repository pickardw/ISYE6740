import numpy as np
import sys
from os.path import abspath, exists
from sklearn.cluster import KMeans
from statistics import mode
from scipy import sparse
import scipy as sp
import matplotlib.pyplot as plt

class KMeans_AI_Compressor:
    def __init__(self):
        self.model_path = abspath("AImodel.npy")
        self.seed = 123
        np.random.seed(123)

    def load_model(self, model_path=None):
        if model_path is None:
            model_path = self.model_path
        model = np.load(model_path)
        return model

    def calculate_size(self, model):
        mem_usage = model.nbytes
        check_usage = sys.getsizeof(model)
        num_idx = model.shape[1]
        num_embeddings = model.shape[0]

        # assume 4 bytes per float32 embedding, 1 byte per uint8 index
        mem_calc = round(((1*num_idx)*(num_embeddings*4)/1000000),4)
        print(f"Model memory usage = {mem_calc} MB")
        return mem_calc

    def calculate_compressed_size(self, indices, centroids):
        num_centroids = centroids.shape[0] 
        centroid_vals = centroids.shape[1] # centroid values are the only float32 values we need
        num_indices = indices.shape[0] # indexes are uint8 values we need to expand to (1000,32)
        mem_calc = round(((num_centroids * centroid_vals * 4 + num_indices * 1)/1000000),4)
        print(f"Compressed memory usage = {mem_calc} MB") 
        return mem_calc

    def normalize_model(self, model):
        norms = np.linalg.norm(model, axis=1, keepdims=True)
        normed = model / norms
        return normed

    def cosine_similarity(self,a,b):
        return np.dot(a,b)/(np.linalg.norm(a,ord=2)*np.linalg.norm(b,ord=2))

    def compress(self, k):
        model = self.load_model()
        original_mem_usage = self.calculate_size(model)
        model = self.normalize_model(model)
        clusterer = KMeans(n_clusters=k, random_state=self.seed)
        cluster_indices = clusterer.fit_predict(model)
        centroids = clusterer.cluster_centers_
        new_embeddings = centroids[cluster_indices] # replace embedding with corresponding centroid val
        sim_values = []
        for i in range(model.shape[0]):
            original_vector = model[i]
            reconstructed_vector = new_embeddings[i]
            cosine_sim = self.cosine_similarity(original_vector,reconstructed_vector)
            sim_values.append(cosine_sim)
        avg_cosine_sim = round(np.mean(sim_values),4)
        print(f"Average cosine similarity = {avg_cosine_sim}")
        compressed_mem_usage = self.calculate_compressed_size(cluster_indices, centroids)
        compression_ratio = round(original_mem_usage/compressed_mem_usage,4) # compression ratio formula found on https://en.wikipedia.org/wiki/Data_compression_ratio
        print(f"Compression ratio = {compression_ratio}")
        return compression_ratio, avg_cosine_sim

if __name__ == "__main__":
    compressor = KMeans_AI_Compressor()
    # for k in [64, 128, 256, 512]:
    for k in [256]:
        result = compressor.compress(k)

    k_vals = [64, 128, 256, 512]
    compression_ratios = []
    avg_cosine_sims = []
    for k in k_vals:
        compression_ratio, avg_cosine_sim = compressor.compress(k)
        compression_ratios.append(compression_ratio)
        avg_cosine_sims.append(avg_cosine_sim)
    
    fig, ax1 = plt.subplots(figsize=(10,6))
    ax1.plot(k_vals, compression_ratios, color='red', label="Compression Ratio")
    ax1.set_ylabel("Compression Ratio", color='red')
    ax1.set_xticks(k_vals)
    ax1.set_xlabel("Number of Centroids (k)")
    ax2 = ax1.twinx()
    ax2.plot(k_vals,avg_cosine_sims,color="blue", label = 'Avg Cosine Similarity')
    ax2.set_ylabel("Avg Cosine Similarity", color='blue')
    plt.show()