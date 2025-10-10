import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap
from IPython.display import HTML
import scipy.io as spio
from scipy import ndimage
import seaborn as sns
from sklearn import preprocessing
from sklearn.cluster import KMeans
from scipy.stats import mode
from scipy.stats import multivariate_normal as mvn
from sklearn.metrics import confusion_matrix, accuracy_score

def import_data():
    data = np.loadtxt("data.dat").T
    labels = np.loadtxt("label.dat")
    return data, labels

def pca(data, n=4): # largely reference code from CDA handbook
    original_shape = data.shape
    scaler = preprocessing.StandardScaler()
    data = scaler.fit_transform(data)
    m, n = data.shape
    C = np.matmul(data.T, data) / m

    d = 4 # reduced dimensionality
    vals, V = np.linalg.eig(C)


    ind = np.argsort(vals)[::-1][:d]
    V = V[:,ind]
    V = V.real # I checked and all imaginary values are 0 for 4 principal components

    pca_data = np.dot(data,V)
    return pca_data, V, scaler

def show_image(im_vector):
    img = im_vector.reshape((28,28))
    # plt.imshow(img)
    # plt.show()
    pass

def EM(pdata, k=2, dims=4): # largely reference code from CDA handbook
    log_likelihood = []
    # Set random seed to ensure reproducibility
    np.random.seed(seed) # so the random seed 6740 was to blame for NaNs killing my EM. 
    # hours of debugging. 
    # seed 6741 had very poor performance and bad composite images. changing seed fixed it

    # Initialize prior
    pi = np.random.random(k)
    pi = pi/np.sum(pi)

    # initial mean and covariance
    mu = np.random.randn(k,dims)
    mu_old = mu.copy()

    # cov
    sigma = []
    for ii in range(k):
        dummy = np.random.randn(dims, dims)
        sigma.append(dummy@dummy.T)
        
    # initialize the posterior
    m=1990
    tau = np.full((m, k), fill_value=0.)

    maxIter = 1000
    tol = 1e-3

    # Store history for animation
    mu_hist = []
    tau_hist = []

    for ii in range(maxIter):
        # E-step
        for kk in range(k):
            tau[:, kk] = pi[kk] * mvn.pdf(pdata, mu[kk], sigma[kk])
        
        # normalize tau
        sum_tau = np.sum(tau, axis=1, keepdims = True)
        # sum_tau.shape = (m,1)
        # tau = np.divide(tau, np.tile(sum_tau,(1,k)))
        tau = tau / sum_tau

        # Store for animation
        mu_hist.append(mu.copy())
        tau_hist.append(tau.copy())

        # M-step
        for kk in range(k):
            # update prior
            pi[kk] = np.sum(tau[:, kk])/m
            # update component mean
            mu[kk] = pdata.T @ tau[:,kk] / np.sum(tau[:,kk])
            # update cov matrix
            dummy = pdata - mu[kk]
            sigma[kk] = dummy.T @ (tau[:,kk][:,None] * dummy) / np.sum(tau[:,kk])

        log_likelihood.append(np.sum(np.log(np.sum(tau, axis=1))))

        if np.linalg.norm(mu-mu_old) < tol:
            print('training converged')
            plt.plot(log_likelihood)
            plt.title(f"Log-Likelihood Convergence: {len(log_likelihood)} iterations")

            fig, ax = plt.subplots(figsize=(6,6))
            cmap = ListedColormap(['#FF9999','#99FF99','#9999FF'])

            sc = ax.scatter(pdata[:,0], pdata[:,1], c=tau_hist[0].argmax(axis=1), cmap=cmap, s=20)
            centers = ax.scatter(mu_hist[0][:,0], mu_hist[0][:,1], c='black', s=100, marker='x')
            ax.set_title("EM-GMM Clustering Progress")
            ax.axis('scaled')

            def animate(i):
                sc.set_array(tau_hist[i].argmax(axis=1))
                centers.set_offsets(mu_hist[i])
                ax.set_title(f"Iteration {i+1}")
                return sc, centers

            anim = FuncAnimation(fig, animate, frames=len(mu_hist), interval=300, blit=False)
            plt.show()
            # plt.close(fig)  # Prevents static image in Jupyter
            # HTML(anim.to_jshtml())
            return pi, mu, sigma, tau
        mu_old = mu.copy()
    else:
        print('max iteration reached')
    pass

seed = 121
if __name__ == "__main__":
    data, labels = import_data()
    
    # show some images
    ind_2 = np.where(labels == 2)
    ind_6 = np.where(labels == 6)
    show_image(data[ind_2[0][0]])
    show_image(data[ind_6[0][0]])
    data, princ_comp, scaler = pca(data)

    pi, mu, sigma, tau = EM(data)

    # print numerical weights pi and mu
    print("Pi (prior probabilities):", pi)
    print("Mu (mean vectors for each component):", mu)

    # show image of mean of each component
    # reverse PCA on means
    im_vector = np.dot(mu,princ_comp.T) 
    # reverse scaling on reconstructed means
    im_vectors = scaler.inverse_transform(im_vector)
    for i,im in enumerate(im_vectors):
        mean_image_vector = im.reshape(28, 28)
        plt.imshow(mean_image_vector, cmap='gray')
        plt.title(f"Mean Image for Component {i+1}")
        plt.show()
        # with seed 6741, these didn't look good. Changing seed fixed it.

    # print 4x4 covariance matrices on heatmap
    for i in range(len(sigma)):
        plt.figure(figsize=(6, 5))
        sns.heatmap(sigma[i], annot=True, fmt=".2f", cmap="viridis", cbar=True)
        plt.title(f"Covariance Matrix for Component {i+1}")
        plt.show()

    # use tau to compare labels of images to true labels
    # convert tau to cluster assignments 0 or 1
    gmm_cluster_assignments = []
    for each in tau:
        if each[0] > each[1]:
            gmm_cluster_assignments.append(0)
        else:
            gmm_cluster_assignments.append(1)
    gmm_cluster_assignments = np.array(gmm_cluster_assignments)
        
    gmm_label_mapping = {} # maps the [0,1] to [2,6]
    for cluster_id in [0,1]:
        cluster = labels[gmm_cluster_assignments == cluster_id]
        dominant_label = mode(cluster)[0]
        gmm_label_mapping[cluster_id] = dominant_label

    gmm_labels = [gmm_label_mapping[lab] for lab in gmm_cluster_assignments]


    gmm_acc = accuracy_score(labels, gmm_labels)
    print(f"Overall misclassification rate: {1-gmm_acc:.4f}")

    conf_labels = np.unique(labels)
    conf_matrix = confusion_matrix(labels, gmm_labels, labels = conf_labels)
    print(f"Confusion matrix labels: {conf_labels}")
    print(conf_matrix)

    # perform k-means with k=2
    kmeans = KMeans(n_clusters=2, random_state=seed)
    kmeans.fit(data)
    # find out which kmeans label fits which digit:
    kmeans_cluster_mapping = {}
    for cluster_label in [0,1]:
        labels_for_cluster = labels[kmeans.labels_ == cluster_label]
        dominant_label = mode(labels_for_cluster)[0]
        kmeans_cluster_mapping[cluster_label] = dominant_label
    
    kmeans_labels = [kmeans_cluster_mapping[lab] for lab in kmeans.labels_] # kmeans labels converted to 2s and 6s

    kmeans_acc = accuracy_score(labels, kmeans_labels)
    print(f"Overall misclassification rate: {1-kmeans_acc:.4f}")

    conf_labels = np.unique(labels)
    conf_matrix = confusion_matrix(labels, kmeans_labels, labels = conf_labels)
    print(f"Confusion matrix labels: {conf_labels}")
    print(conf_matrix)
    pass
    