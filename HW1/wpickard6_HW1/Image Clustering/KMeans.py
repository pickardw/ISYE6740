from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from os.path import abspath
import time 

class KMeansImpl:
    def __init__(self):
        ## TODO add any params to be needed for the clustering algorithm.
        self.k_list = [3,6,12,24,48]
        self.image_files = ["big_yosh.png", "parrots.png", "football.bmp"]
        np.random.seed(123)
        pass

    def load_image(self, image_name="big_yosh.png"):
        """
        Returns the image numpy array.
        It is important that image_name parameter defaults to the choice image name.
        """
        image_fp = abspath(f'{image_name}')
        return np.array(Image.open(image_fp))

    def flatten_matrix(self, img):
        self.original_shape = img.shape
        # print("image shape: ", img.shape)
        flattened = img.reshape(-1, img.shape[-1]) # each row contains 3 columns (r,g,b)
        # print("flattened shape: ", flattened.shape)
        return flattened

    def init_centroids(self, pixels, num_clusters):
        flattened_pixels = self.flatten_matrix(pixels)
        
        # ensure unique centroid colors - no duplicate RGB values
        unique_pixels = np.unique(flattened_pixels, axis=0)
        random_indices = np.random.choice(unique_pixels.shape[0], size=num_clusters, replace=False)
        centroids = unique_pixels[random_indices]
        return centroids
    
    def calculate_distance(self, pixels, centroids, norm_distance):
        # fix data type overflow
        pixels = pixels.astype(np.float64).T # matching shape convention from demo code
        centroids = centroids.astype(np.float64).T # matching shape convention from demo code

        
        if norm_distance == 1:
            # use broadcasting to enable vectorized distance calculation
            pixels_broadcasted = pixels.T[:,np.newaxis,:]  # shape (num_pixels, 1, 3)
            centroids_broadcasted = centroids.T[np.newaxis, :, :]     # shape (1, num_centroids, 3)
            distances = np.sum(np.abs(pixels_broadcasted - centroids_broadcasted), axis=2)
        elif norm_distance == 2:
            c_l2norm_squared = np.sum(np.power(centroids,2), axis=0,keepdims=True)
            distances = (2*np.dot(pixels.T,centroids) - c_l2norm_squared)
            # distances_linalg = np.linalg.norm(pixels[:, np.newaxis] - centroids,ord =2, axis=2)
            # distances = np.sqrt(np.sum((pixels_broadcasted - centroids_broadcasted)**2, axis=2))

            # used to verify correctness when switching to vectorized calculation
            # vector_labels = np.argmax(distances_better, axis=1)
            # norm_labels = np.argmin(distances, axis=1)
        return distances

    def reconstruct_image(self, cluster_assignments, centroids):
        reconstructed = centroids[cluster_assignments].reshape(self.original_shape)
        return reconstructed
    
    def compress(self, pixels, num_clusters, norm_distance=2):
        """
        Compress the image using K-Means clustering.

        Parameters:
            pixels: 3D image for each channel (a, b, 3), values range from 0 to 255.
            num_clusters: Number of clusters (k) to use for compression.
            norm_distance: Type of distance metric to use for clustering.
                            Can be 1 for Manhattan distance or 2 for Euclidean distance.
                            Default is 2 (Euclidean).

        Returns:
            Dictionary containing:
                "class": Cluster assignments for each pixel.
                "centroid": Locations of the cluster centroids.
                "img": Compressed image with each pixel assigned to its closest cluster.
                "number_of_iterations": total iterations taken by algorithm
                "time_taken": time taken by the compression algorithm
        """
        map = {
            "class": None,
            "centroid": None,
            "img": None,
            "number_of_iterations": None,
            "time_taken": None,
            "additional_args": {}
        }

        '''
        # TODO - Add your implementation here.
        '''
        start = time.time()
        centroids = self.init_centroids(pixels, num_clusters)
        pixels = self.flatten_matrix(pixels)
        for i in range(1000): # max iterations
            # print(f"Iteration {i+1}")

            distances = self.calculate_distance(pixels, centroids, norm_distance)
            if norm_distance == 2:
                cluster_assignments = np.argmax(distances, axis=1)
            elif norm_distance == 1:
                cluster_assignments = np.argmin(distances, axis=1)
            
            old_centroids = centroids.copy()
            empty_centroid_indices = []
            wcss = 0.0 # use wcss to pick best k
            for k in range(num_clusters):
                cluster_pixels = pixels[cluster_assignments == k]
                if len(cluster_pixels) > 0:
                    centroids[k] = np.mean(cluster_pixels, axis=0)
                    wcss += np.sum((cluster_pixels - centroids[k])**2)
                else:
                    empty_centroid_indices.append(k)
            
            # remove empty centroids
            if empty_centroid_indices:
                centroids = np.delete(centroids, empty_centroid_indices, axis=0)
                num_clusters = len(centroids)
                print(f"Removed {len(empty_centroid_indices)} empty clusters. New cluster count: {num_clusters}")
                continue # skip convergence check if centroids were removed

            # check for convergence
            if np.allclose(old_centroids, centroids):
                # print(f"Converged in {i+1} iterations.")
                map["additional_args"]["final_cluster_count"] = len(centroids)
                map["additional_args"]["wcss"] = wcss
                map["number_of_iterations"] = i + 1
                map["class"] = cluster_assignments
                map["centroid"] = centroids
                map["img"] = self.reconstruct_image(cluster_assignments, centroids)
                # compressed = Image.fromarray(map["img"])
                # compressed.show()
                break

        
        end = time.time()
        map["time_taken"] = end - start

        return map

def save_elbow_plot(results, title):
    k_values = [res["additional_args"]["final_cluster_count"] for res in results]
    wcss_values = [res["additional_args"]["wcss"] for res in results]

    plt.figure(figsize=(8,5))
    plt.plot(k_values, wcss_values, marker='o')
    plt.title(title)
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
    plt.xticks(k_values)
    plt.grid(True)
    plt.savefig(title.replace(" ", "_") + ".png")
    plt.close()

def save_image(image_array, filename):
    img = Image.fromarray(image_array)
    img.save(filename)

if __name__ == "__main__":
    save_result = False
    start = time.time()
    kmeans = KMeansImpl()
    for image_name in kmeans.image_files:
        l2_results = []
        image_paths = []
        image = kmeans.load_image(image_name)
        for cluster_count in kmeans.k_list:
            result = kmeans.compress(image, num_clusters=cluster_count, norm_distance=2)
            print(f"{image_name} | k={cluster_count} | i={result['number_of_iterations']} | t={result['time_taken']}")
            if save_result:
                l2_results.append(result)
                output_filename = abspath(f"results/{image_name.split('.')[0]}_k{cluster_count}_norm2.{image_name.split('.')[-1]}")
                if output_filename.split(".")[-1] == "bmp":
                    output_filename = output_filename.replace(".bmp", ".png")
                image_paths.append(output_filename)
                save_image(result["img"], output_filename)
        if save_result:
            save_elbow_plot(l2_results, f"results/{image_name} L2 Elbow Plot")

    end = time.time()
    print(f"Total time taken: {(end-start)/60} minutes")

    start = time.time()
    kmeans = KMeansImpl()
    for image_name in kmeans.image_files:
        l1_results = []
        image = kmeans.load_image(image_name)
        for cluster_count in kmeans.k_list:
            result = kmeans.compress(image, num_clusters=cluster_count, norm_distance=1)
            print(f"{image_name} | k={cluster_count} | i={result['number_of_iterations']} | t={result['time_taken']}")
            if save_result:
                l1_results.append(result)
                output_filename = abspath(f"results/{image_name.split('.')[0]}_k{cluster_count}_norm1.{image_name.split('.')[-1]}")
                if output_filename.split(".")[-1] == "bmp":
                    output_filename = output_filename.replace(".bmp", ".png") # overleaf doesn't take bmp files
                save_image(result["img"], output_filename)
        if save_result:
            save_elbow_plot(l1_results, f"results/{image_name} L1 Elbow Plot")

    end = time.time()
    print(f"Total time taken: {(end-start)/60} minutes")
    # compressed_img = Image.fromarray(result["img"].astype(np.uint8))
    # compressed_img.save("compressed_image.jpeg")
