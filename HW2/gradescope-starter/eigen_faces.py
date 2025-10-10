# -----------------------------------------------------------------------------
# NOTE: This file consists of 2 classes

# 1. EigenFacesResult - This class should not be modified. Gradescope will use the output of run() 
# method in this format.
# 2. EigenFaces - This is class which will implement the eigen faces algorithm and return the results.  
# -----------------------------------------------------------------------------

import os
import numpy as np
import matplotlib.pyplot as plt
import sklearn.preprocessing as skpp
from PIL import Image

# -----------------------------------------------------------------------------
# NOTE: This class should NOT be modified.
# Gradescope will depend on the structure of this class as defined. 
# -----------------------------------------------------------------------------
class EigenFacesResult:
    """    
    A structured container for storing the results of the EigenFaces computation.

    Attributes
    ----------
    subject_1_eigen_faces : np.ndarray
        A (6, a, b) array representing the top 6 eigenfaces for subject 1.
        A plt.imshow(map['subject_1_eigen_faces'][0]) should display first in a eigen face for subject 1

    subject_2_eigen_faces : np.ndarray
        A (6, a, b) array representing the top 6 eigenfaces for subject 2.
        A plt.imshow(map['subject_2_eigen_faces'][0]) should display first in a eigen face for subject 2

    s11 : float
        Projection residual of subject 1 test image on subject 1 eigenfaces.

    s12 : float
        Projection residual of subject 2 test image on subject 1 eigenfaces.

    s21 : float
        Projection residual of subject 1 test image on subject 2 eigenfaces.

    s22 : float
        Projection residual of subject 2 test image on subject 2 eigenfaces.
    """

    def __init__(
        self,
        subject_1_eigen_faces: np.ndarray,
        subject_2_eigen_faces: np.ndarray,
        s11: float,
        s12: float,
        s21: float,
        s22: float
    ):
        self.subject_1_eigen_faces = subject_1_eigen_faces
        self.subject_2_eigen_faces = subject_2_eigen_faces
        self.s11 = s11
        self.s12 = s12
        self.s21 = s21
        self.s22 = s22
        
# -----------------------------------------------------------------------------
# NOTE: Do not change the parameters / return types for pre defined methods.
# -----------------------------------------------------------------------------
class EigenFaces:
    """
    This class handles loading facial images for two subjects, computing eigenfaces
    via PCA, and evaluating projection residuals for test images.

    Methods
    -------
    run():
        Computes the eigenfaces for each subject and the projection residuals for test images.
    """

    def __init__(self, images_root_directory="data/yalefaces"):
        """
        Initializes the EigenFaces object and loads all relevant facial images from the specified directory.

        Parameters
        ----------
        images_root_directory : str
            The path to the root directory containing subject images.
        """
        self.original_sizes = (320,243)
        self.downsampled_sizes = (80,60)
        gif_files = [f"{images_root_directory}/{f}" for f in os.listdir(images_root_directory)]

        self.s1_train_files = sorted([f for f in gif_files if 'subject01' in f and 'test' not in f])
        self.s2_train_files = sorted([f for f in gif_files if 'subject02' in f and 'test' not in f])
        self.s1_test_file = f'{images_root_directory}/subject01-test.gif'
        self.s2_test_file = f'{images_root_directory}/subject02-test.gif'
        self.s1_train_images = np.array([self.load_image(file) for file in self.s1_train_files])
        self.s2_train_images = np.array([self.load_image(file) for file in self.s2_train_files])
        self.s1_test_image = np.array(self.load_image(self.s1_test_file))
        self.s2_test_image = np.array(self.load_image(self.s2_test_file))

    def load_image(self, file, ds_factor: int = 4) -> np.ndarray:
            with Image.open(file) as img:
                downsampled = img.resize((img.size[0]//ds_factor, img.size[1]//ds_factor), Image.LANCZOS)
                ds_array = np.array(downsampled)
            return ds_array.flatten()

    def compute_eigenfaces(self, images, num_eigenfaces = 6):
        scaler = skpp.StandardScaler(with_std=False).fit(images)
        matrix = scaler.transform(images)
        U, sigma, VT = np.linalg.svd(matrix)
        eigenfaces_flat = VT[:num_eigenfaces] # SVD returns values in descending order of significance
        eigenfaces = VT[:num_eigenfaces].reshape((num_eigenfaces, self.downsampled_sizes[1], self.downsampled_sizes[0]))
        return eigenfaces, eigenfaces_flat, scaler.mean_
    
    def compute_residual(self, test_image, eigenfaces_flat, eigen_mean):
        center_test = test_image - eigen_mean
        s = center_test - eigenfaces_flat.T @ (eigenfaces_flat @ center_test)
        return np.linalg.norm(s)**2
    
    def save_eigenfaces_gridplot(self, eigenfaces, subject):
        fig, axs = plt.subplots(2,3)
        axs = axs.flatten()
        for i, axis in enumerate(axs):
            axis.imshow(eigenfaces[i],cmap="gray")
            axis.set_title(f"Subject {subject} Eigenface {i+1}")
            axis.axis("off")
        plt.tight_layout()
        plt.savefig(f"eigenfaces_subject{subject}.png")

    def run(self) -> EigenFacesResult:
        """
        Computes eigenfaces for both subjects and projection residuals
        for test images using those eigenfaces.

        Returns
        -------
        EigenFacesResult
            Object containing eigenfaces and residuals for both subjects.
        """
        # find eigenfaces for subjects and save grid plots of images
        eigenfaces_1, s1_flatfaces, s1_mean = self.compute_eigenfaces(self.s1_train_images)
        self.save_eigenfaces_gridplot(eigenfaces_1, subject=1)
        eigenfaces_2, s2_flatfaces, s2_mean = self.compute_eigenfaces(self.s2_train_images)
        self.save_eigenfaces_gridplot(eigenfaces_2, subject=2)

        # compute residuals
        projection_residual_s11 = self.compute_residual(self.s1_test_image, s1_flatfaces, s1_mean)
        projection_residual_s12 = self.compute_residual(self.s2_test_image, s1_flatfaces, s1_mean)
        projection_residual_s21 = self.compute_residual(self.s1_test_image, s2_flatfaces, s2_mean)
        projection_residual_s22 = self.compute_residual(self.s2_test_image, s2_flatfaces, s2_mean)


        return EigenFacesResult(
            subject_1_eigen_faces=eigenfaces_1,
            subject_2_eigen_faces=eigenfaces_2,
            s11=projection_residual_s11,
            s12=projection_residual_s12,
            s21=projection_residual_s21,
            s22=projection_residual_s22
        )
    
if __name__ == "__main__":
    ef = EigenFaces()
    result = ef.run()
    print(f"Subject 1 Test Residuals: \n\t S1 Eigenfaces: {result.s11}\n\t S2 Eigenfaces: {result.s21}")
    print(f"Subject 2 Test Residuals: \n\t S1 Eigenfaces: {result.s12}\n\t S2 Eigenfaces: {result.s22}")
    pass