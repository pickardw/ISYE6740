# -----------------------------------------------------------------------------
# NOTE: This file consists of 2 classes

# 1. EigenFacesResult - This class should not be modified. Gradescope will use the output of run() 
# method in this format.
# 2. EigenFaces - This is class which will implement the eigen faces algorithm and return the results.  
# -----------------------------------------------------------------------------



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
        raise NotImplementedError("Not Implemented")
        
    def run(self) -> EigenFacesResult:
        """
        Computes eigenfaces for both subjects and projection residuals
        for test images using those eigenfaces.

        Returns
        -------
        EigenFacesResult
            Object containing eigenfaces and residuals for both subjects.
        """
        return EigenFacesResult(
            subject_1_eigen_faces=eigenfaces_1,
            subject_2_eigen_faces=eigenfaces_2,
            s11=projection_residual_s11,
            s12=projection_residual_s12,
            s21=projection_residual_s21,
            s22=projection_residual_s22
        )