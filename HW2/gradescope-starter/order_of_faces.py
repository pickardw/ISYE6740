import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import scipy.io as spio
from scipy.sparse.csgraph import connected_components
from scipy.sparse.csgraph import shortest_path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# -----------------------------------------------------------------------------
# NOTE: Do not change the parameters / return types for pre defined methods.
# -----------------------------------------------------------------------------

class OrderOfFaces:
    """
    This class handles loading and processing facial image data for dimensionality
    reduction using the ISOMAP algorithm, with PCA as an optional comparison.

    Attributes:
    ----------
    images_path : str
        Path to the .mat file containing the image dataset.

    Methods:
    -------
    get_adjacency_matrix(epsilon):
        Returns the adjacency matrix based on a given epsilon neighborhood.

    get_best_epsilon():
        Returns the best epsilon for the ISOMAP algorithm, likely based on
        graph connectivity or reconstruction error.

    isomap(epsilon):
        Computes a 2D embedding of the data using the ISOMAP algorithm.

    pca(num_dim):
        Returns a low-dimensional embedding of the data using PCA.
    """

    def __init__(self, images_path='data/isomap.mat'):
        """
        Initializes the OrderOfFaces object and loads image data from the given path.

        Parameters:
        ----------
        images_path : str
            Path to the .mat file containing the facial images dataset.
        """
        self.visualizations = False
        self.images = spio.loadmat(images_path,squeeze_me=True)["images"] # each column an image: (4096,698) - known 698 images
        self.images = self.images.T # make every row an image to match question
        self.m, self.n =self.images.shape
        np.random.seed(6740)
        self.get_distance_matrix()

    def get_distance_matrix(self) -> np.ndarray:
        # compute pairwise distances
        distances = np.zeros((self.m, self.m))
        for i in range(self.m):
            for j in range(i + 1, self.m):
                dist = np.linalg.norm(self.images[i, :] - self.images[j, :])
                distances[i, j] = dist
        distances = distances + distances.T
        
        self.distances = distances

    def plot_adjacency_matrix(self, data):
        zero_indices = np.argwhere(data == 0)
        # pick 5 random indexes from zero_indices
        sample_zero_indices = zero_indices[np.random.choice(zero_indices.shape[0], 5, replace=False)]
        non_neighbor_images = []
        for idx_pairs in sample_zero_indices:
            i, j = idx_pairs[0], idx_pairs[1]
            im1 = self.images[i, :].reshape(64, 64).T
            im2 = self.images[j, :].reshape(64, 64).T
            non_neighbor_images.append((i,j,im1, im2))
        
        nonzero_indices = np.argwhere(data != 0)
        sample_nonzero_indices = nonzero_indices[np.random.choice(nonzero_indices.shape[0],5)]
        neighbor_images = []
        for idx_pairs in sample_nonzero_indices:
            i, j = idx_pairs[0], idx_pairs[1]
            im1 = self.images[i, :].reshape(64, 64).T
            im2 = self.images[j, :].reshape(64, 64).T
            neighbor_images.append((i,j,im1, im2))
        
        fig, axs = plt.subplots(figsize=(12,12))
        # reference for placing images: https://stackoverflow.com/questions/4860417/placing-custom-images-in-a-plot-window-as-custom-data-markers-or-to-annotate-t/4872190
        
        im = axs.imshow(data, vmin=1, cmap='viridis', aspect='auto')
        fig.colorbar(im, ax=axs, label='Euclidean Distance')
        axs.set_title("Adjacency Matrix Heatmap")
        axs.set_xlabel("Data Point Index")
        axs.set_ylabel("Data Point Index")
        for i,j,im1,im2 in non_neighbor_images:
            imbox1 = OffsetImage(im1, zoom=0.5, cmap='gray',label=f'Img {i}')
            abox1 = AnnotationBbox(imbox1, (i,j), xybox=(-20,20), xycoords='data',arrowprops={'arrowstyle':"->"}, bboxprops={'boxstyle':"round","fc":"red"} ,boxcoords="offset points")
            axs.add_artist(abox1)
            imbox2 = OffsetImage(im2, zoom=0.5, cmap='gray', label=f'Im {j}')
            abox2 = AnnotationBbox(imbox2, (i,j), xybox=(20,20), xycoords='data',arrowprops={'arrowstyle':"->"}, bboxprops={'boxstyle':"round","fc":"red"} ,boxcoords="offset points")
            axs.add_artist(abox2)
        for i,j,im1,im2 in neighbor_images:
            imbox1 = OffsetImage(im1, zoom=0.5, cmap='gray',label=f'Im {i}')
            abox1 = AnnotationBbox(imbox1, (i,j), xybox=(-20,20), xycoords='data',arrowprops={'arrowstyle':"->"}, bboxprops={'boxstyle':"square","fc":"blue"} ,boxcoords="offset points")
            axs.add_artist(abox1)
            imbox2 = OffsetImage(im2, zoom=0.5, cmap='gray',label=f'Im {j}')
            abox2 = AnnotationBbox(imbox2, (i,j), xybox=(20,20), xycoords='data',arrowprops={'arrowstyle':"->"}, bboxprops={'boxstyle':"square","fc":"blue"} ,boxcoords="offset points")
            axs.add_artist(abox2)
        plt.show()
    
    def get_adjacency_matrix(self, epsilon: float) -> np.ndarray:
        """
        Constructs the adjacency matrix using epsilon neighborhoods.

        Parameters:
        ----------
        epsilon : float
            The neighborhood radius within which points are considered connected.

        Returns:
        -------
        np.ndarray
            A 2D adjacency matrix (m x m) where each entry represents distance between
            neighbors within the epsilon threshold.
        """
        
        # construct adjacency matrix
        adj_matrix = self.distances.copy()
        adj_matrix[adj_matrix > epsilon] = 0

        return adj_matrix

    def get_best_epsilon(self) -> float:
        """
        Heuristically determines the best epsilon value for graph connectivity in ISOMAP.

        Returns:
        -------
        float
            Optimal epsilon value ensuring a well-connected neighborhood graph.
        """
  
        # going to find an epsilon that allows for a certain number of average neighbors - https://stackoverflow.com/questions/43763362/obtaining-the-size-of-the-neighborhood-in-isomap-algorithm
        min_distance = np.min(self.distances[self.distances > 0])
        max_distance = np.max(self.distances)
        epsilon_list = np.linspace(min_distance, max_distance / 2, 20)

        best_connecting_epsilon = None
        target_avg_degree = 8 #shot in the dark based on stackoverflow post
        for epsilon in epsilon_list:
            adj_matrix = self.get_adjacency_matrix(epsilon)
            n_components, _ = connected_components(csgraph=adj_matrix, directed=False, return_labels=True)

            print(f"testing epsilon {epsilon:.2f}: {n_components}.")
            if n_components == 1:
                avg_degree = np.count_nonzero(adj_matrix) / self.m
                print(f"\t average degree: {avg_degree:.2f}")

                if best_connecting_epsilon == None:
                    best_connecting_epsilon = epsilon # pick first epsilon that connects graph. avg degrees looked fine
                    if self.visualizations:
                        self.plot_adjacency_matrix(adj_matrix)
                    return epsilon


    def isomap(self, epsilon: float) -> np.ndarray:
        """
        Applies the ISOMAP algorithm to compute a 2D low-dimensional embedding of the dataset.

        Parameters:
        ----------
        epsilon : float
            The neighborhood radius for building the adjacency graph.

        Returns:
        -------
        np.ndarray
            A (m x 2) array where each row is a 2D embedding of the original data point.
        """
        adj_matrix = self.get_adjacency_matrix(epsilon)
        

        
        dist_matrix = shortest_path(adj_matrix, directed=False, method='auto')
        # centering matrix
        H = np.eye(self.m) - np.ones((self.m,self.m)) / self.m
        C = -0.5 * H @ (dist_matrix**2) @ H

        # get eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(C)

        # sorting eigens https://stackoverflow.com/questions/8092920/sort-eigenvalues-and-associated-eigenvectors-after-using-numpy-linalg-eig-in-pyt
        sort_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sort_indices]
        eigenvectors = eigenvectors[:, sort_indices]

        # compute the reduced representation with 2 dimensions
        lambda_sqrt = np.diag(np.sqrt(eigenvalues[:2]))
        U = eigenvectors[:, :2]
        Z = U @ lambda_sqrt
        
        if self.visualizations:
            fig, axs = plt.subplots(figsize=(10, 10))
            axs.scatter(Z[:, 0], Z[:, 1])
            axs.set_title('ISOMAP 2D Embedding')
            axs.set_xlabel('ISOMAP Dimension 1')
            axs.set_ylabel('ISOMAP Dimension 2')

            # pick 20 images to display
            sample_indices = np.random.choice(self.m,50,replace=False)
            for i in sample_indices:
                x = Z[i, 0]
                y = Z[i, 1]
                img = self.images[i,:].reshape(64,64).T
                imbox1 = OffsetImage(img, zoom=0.5, cmap='gray',label=f'Img {i}')
                abox1 = AnnotationBbox(imbox1, (x,y), xybox=(-20,20), xycoords='data',arrowprops={'arrowstyle':"->"}, bboxprops={'boxstyle':"round","fc":"red"} ,boxcoords="offset points")
                axs.add_artist(abox1)
            plt.show()
        return Z

    def pca(self, num_dim: int) -> np.ndarray:
        """
        Applies PCA to reduce the dataset to a specified number of dimensions.

        Parameters:
        ----------
        num_dim : int
            Number of principal components to project the data onto.

        Returns:
        -------
        np.ndarray
            A (m x num_dim) array representing the dataset in a reduced PCA space.
        """
        scaler = StandardScaler()
        scaled_im = scaler.fit_transform(self.images)
        pca = PCA(n_components=num_dim).fit_transform(scaled_im)
        if self.visualizations:
            fig, axs = plt.subplots(figsize=(10, 10))
            axs.scatter(pca[:, 0], pca[:, 1])
            axs.set_title('PCA 2D Embedding')
            axs.set_xlabel('Principal Component 1')
            axs.set_ylabel('Principal Component 2')

            # pick 20 images to display
            sample_indices = np.random.choice(self.m,50,replace=False)
            for i in sample_indices:
                x = pca[i, 0]
                y = pca[i, 1]
                img = self.images[i,:].reshape(64,64).T
                imbox1 = OffsetImage(img, zoom=0.5, cmap='gray',label=f'Img {i}')
                abox1 = AnnotationBbox(imbox1, (x,y), xybox=(-20,20), xycoords='data',arrowprops={'arrowstyle':"->"}, bboxprops={'boxstyle':"round","fc":"red"} ,boxcoords="offset points")
                axs.add_artist(abox1)
            plt.show()
        return np.array(pca)

if __name__ == "__main__":
    of = OrderOfFaces()
    of.visualizations = True
    eps = of.get_best_epsilon()
    of.isomap(eps)
    of.pca(2)