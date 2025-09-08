import numpy as np
from os.path import abspath, exists
from sklearn.cluster import KMeans
from statistics import mode
from scipy import sparse
import scipy as sp

class PoliticalBlogsClustering:
    def __init__(self):
        edges_fp = abspath("data/edges.txt")
        nodes_fp = abspath("data/nodes.txt")
        self.seed = 123
        np.random.seed(123)

        self.node_info = []
        with open(nodes_fp, 'r') as f:
            for line in f.readlines():
                index, blog, label, _ = line.split("\t")
                self.node_info.append((blog, int(label)))\
        
        with open(edges_fp, 'r') as f:
            lines = [line.split() for line in f]
        self.edges = np.array(lines)
        pass

    def preprocess_graph(self, A):
        degrees = np.array(A.sum(axis=1)).flatten()
        isolation_mask = degrees < 1
        isolated_nodes = np.where(isolation_mask)[0]
        keep_nodes = np.where(~isolation_mask)[0]
        # print(f"Found {len(isolated_nodes)} isolated nodes")

        # remove isolated nodes from adjacency matrix
        A = A.toarray()
        A = np.delete(A, isolated_nodes, axis = 1)
        A = np.delete(A, isolated_nodes, axis = 0)
        return A, isolated_nodes, keep_nodes
    
    def compute_mismatch(self,cluster_labels):
        majority_label = mode(cluster_labels)
        mismatches = sum(1 for label in cluster_labels if label != majority_label)
        # print(f"Mismatch rate: {mismatches}/{len(cluster_labels)}")
        mismatch_rate = (mismatches / len(cluster_labels))
        return majority_label, mismatches, round(mismatch_rate, 2)
    
    def find_majority_labels(self, num_clusters = 2):
        '''
        This method loads the data, performs spectral clustering  and reports the majority labels

        Inputs:
            num_clusters (int): The number of clusters to be created

        Output:
            A map with following attributes
            1. overall_mismatch_rate: <2 decimal places>
            2. mismatch_rates: [{"majority_index": <int>, "mismatch_rate": <2 decimal places>}]
        '''

        map = {
            "overall_mismatch_rate": 0.0,
            "mismatch_rates": [],
        }

        # TODO - start your implementation.
        ## It is suggested to break your code into smaller methods but not nessasary. 
        n1 = self.edges[:,0].astype(int)-1
        n2 = self.edges[:,1].astype(int)-1 # decrement indexes to be 0-based
        n = len(self.node_info)
        v = np.ones((self.edges.shape[0],1)).flatten()
        A = sparse.coo_matrix((v, (n1, n2)), shape=(n,n))
        A = A + A.T # makes adjacency matrix symmetric = graph undirected
        check_A = A.copy().toarray()
        
        A, isolated_nodes, keep_nodes = self.preprocess_graph(A)
        A = sparse.csc_matrix(A) # dense matrix with no isolated nodes

        D = np.diag(1/np.sqrt(np.sum(A, axis=1)).A1) # calculates inverse square root degree matrix - for normalized symmetric graph laplacian
        L = np.array(D @ A @ D)

        v, x = np.linalg.eig(L) # eigen decomposition
        idx_sorted = np.argsort(v)
        x = x[:, idx_sorted[-num_clusters:]] # selects k largest eigenvectors
        x = x/np.repeat(np.sqrt(np.sum(x*x, axis=1).reshape(-1,1)), num_clusters, axis=1)
        
        kmeans = KMeans(n_clusters=num_clusters, random_state=self.seed).fit(x.real)
        cluster_indices = kmeans.labels_
        for cluster in range(num_clusters):
            # print(f"Cluster {cluster+1}")
            cluster_labels = []
            for index, assignment in enumerate(cluster_indices):
                if assignment == cluster:
                    blog_index = keep_nodes[index] # original index was destroyed during isolation preprocessing
                    blog, label = self.node_info[blog_index]
                    cluster_labels.append(label)
                    # print(f"{blog}")
            majority_label, mismatch_count, mismatch_rate = self.compute_mismatch(cluster_labels)
            map["overall_mismatch_rate"] += mismatch_count
            map["mismatch_rates"].append({"majority_index": int(majority_label), "mismatch_rate": mismatch_rate})
        map["overall_mismatch_rate"] = round((map["overall_mismatch_rate"]/len(keep_nodes)), 2)
        return map

if __name__ == "__main__":
    spec_clust = PoliticalBlogsClustering()
    results = {}
    for k in [2,5,10,30,50]:
        results[k] = spec_clust.find_majority_labels(num_clusters = k)
    for k, result in results.items():
        print(f"\t k= {k} overall mismatch rate = {result['overall_mismatch_rate']}%")
        for rate in result['mismatch_rates']:
            print(f"{rate}%")

    # tune k within best range above
    results = {}
    for k in range(2,20):
        results[k] = spec_clust.find_majority_labels(num_clusters = k)
    for k, result in results.items():
        print(f"{k}, {result['overall_mismatch_rate']}%")
        for rate in result['mismatch_rates']:
            print(f"{rate['mismatch_rate']}%")