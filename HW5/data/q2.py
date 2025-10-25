import numpy as np
from scipy.io import loadmat
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd

# Set random seed for reproducibility
RANDOM_STATE = 6740

def load_and_preprocess_data(filepath):
    """
    Load MNIST data and standardize from [0,255] to [0,1]
    
    Analytics Note: Standardization ensures all features contribute equally.
    Without it, algorithms using distance metrics (KNN, SVM) or gradient 
    descent (Neural Networks, Logistic Regression) may struggle.
    """
    data = loadmat(filepath)
    
    # TODO: Extract xtrain, ytrain, xtest, ytest from the loaded data
    # Hint: Check what keys are in the .mat file
    xtrain = data['???']  # Shape should be (60000, 784)
    ytrain = data['???']  # Shape should be (60000,)
    xtest = data['???']   # Shape should be (10000, 784)
    ytest = data['???']   # Shape should be (10000,)
    
    # TODO: Standardize the features from [0, 255] to [0, 1]
    # Hint: Simple division will work here
    xtrain = xtrain / ???
    xtest = xtest / ???
    
    # Flatten labels if needed (sometimes they're (n,1) instead of (n,))
    ytrain = ytrain.ravel()
    ytest = ytest.ravel()
    
    print(f"Data shapes: xtrain={xtrain.shape}, ytrain={ytrain.shape}")
    print(f"             xtest={xtest.shape}, ytest={ytest.shape}")
    
    return xtrain, ytrain, xtest, ytest


def downsample_for_svm(xtrain, ytrain, m=5000, random_state=RANDOM_STATE):
    """
    Downsample training data for SVM efficiency.
    
    Analytics Note: SVMs have O(n²) to O(n³) complexity. With 60k samples,
    training takes too long. We sacrifice some accuracy for practicality.
    """
    np.random.seed(random_state)
    indices = np.random.choice(len(xtrain), size=m, replace=False)
    return xtrain[indices], ytrain[indices]


def train_classifiers(xtrain, ytrain, dataset_name):
    """
    Train all 5 classifiers and return them in a dictionary.
    
    Analytics Note: Each classifier has different strengths:
    - KNN: Simple, non-parametric, no training needed (lazy learner)
    - Logistic Regression: Fast, interpretable, works well for linearly separable data
    - Linear SVM: Similar to logistic regression but with margin maximization
    - Kernel SVM (RBF): Can capture non-linear patterns via kernel trick
    - Neural Network: Most flexible, can learn complex hierarchical features
    """
    classifiers = {}
    
    # 1. KNN Classifier
    print(f"\n[{dataset_name}] Training KNN...")
    # TODO: Experiment with K values. Start with k=3 or k=5
    # Question: Why might K=1 overfit? Why might K=60000 underfit?
    classifiers['KNN'] = KNeighborsClassifier(n_neighbors=???)
    classifiers['KNN'].fit(xtrain, ytrain)
    
    # 2. Logistic Regression
    print(f"[{dataset_name}] Training Logistic Regression...")
    # TODO: Create LogisticRegression with max_iter=1000, random_state=RANDOM_STATE
    # Use multi_class='multinomial' and solver='lbfgs' for multi-class
    classifiers['Logistic Regression'] = LogisticRegression(
        max_iter=???,
        random_state=RANDOM_STATE,
        multi_class='???',
        solver='???'
    )
    classifiers['Logistic Regression'].fit(xtrain, ytrain)
    
    # 3. Linear SVM
    print(f"[{dataset_name}] Training Linear SVM...")
    # Downsample for efficiency
    xtrain_svm, ytrain_svm = downsample_for_svm(xtrain, ytrain)
    # TODO: Create SVC with kernel='linear', random_state=RANDOM_STATE
    classifiers['SVM (Linear)'] = SVC(kernel='???', random_state=RANDOM_STATE)
    classifiers['SVM (Linear)'].fit(xtrain_svm, ytrain_svm)
    
    # 4. Kernel SVM (RBF)
    print(f"[{dataset_name}] Training Kernel SVM (RBF)...")
    # TODO: Create SVC with kernel='rbf', random_state=RANDOM_STATE
    # Analytics Note: RBF kernel computes similarity using Gaussian functions
    # This allows non-linear decision boundaries without explicit feature mapping
    classifiers['Kernel SVM (RBF)'] = SVC(kernel='???', random_state=RANDOM_STATE)
    classifiers['Kernel SVM (RBF)'].fit(xtrain_svm, ytrain_svm)
    
    # 5. Neural Network (MLP)
    print(f"[{dataset_name}] Training Neural Network...")
    # TODO: Create MLPClassifier with hidden_layer_sizes=(20, 10)
    # Use max_iter=300, random_state=RANDOM_STATE
    # Analytics Note: (20, 10) means 2 hidden layers with 20 and 10 neurons
    # This creates a hierarchy: 784 → 20 → 10 → 10 (output classes)
    classifiers['Neural Network'] = MLPClassifier(
        hidden_layer_sizes=(???, ???),
        max_iter=???,
        random_state=RANDOM_STATE
    )
    classifiers['Neural Network'].fit(xtrain, ytrain)
    
    return classifiers


def evaluate_classifiers(classifiers, xtest, ytest, dataset_name):
    """
    Evaluate all classifiers and return detailed metrics.
    
    Analytics Note: 
    - Precision: Of all predicted class X, how many were correct? (TP / (TP + FP))
    - Recall: Of all actual class X, how many did we find? (TP / (TP + FN))
    - F1-Score: Harmonic mean of precision and recall (2 * P * R / (P + R))
    """
    results = {}
    
    for clf_name, clf in classifiers.items():
        print(f"\n[{dataset_name}] Evaluating {clf_name}...")
        
        # Make predictions
        y_pred = clf.predict(xtest)
        
        # TODO: Calculate precision, recall, and F1-score for each class
        # Hint: Use precision_recall_fscore_support with appropriate parameters
        # Set average=None to get per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            ytest, 
            y_pred, 
            average=???,  # None for per-class metrics
            zero_division=0
        )
        
        # Store results as a DataFrame for easy viewing
        results[clf_name] = pd.DataFrame({
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'Support': support
        })
        results[clf_name].index.name = 'Class'
        
        # Calculate and display overall accuracy
        accuracy = (y_pred == ytest).mean()
        print(f"{clf_name} Accuracy: {accuracy:.4f}")
    
    return results


def display_results(results, dataset_name):
    """Display results in a clear, organized format."""
    print(f"\n{'='*80}")
    print(f"RESULTS FOR {dataset_name.upper()}")
    print(f"{'='*80}")
    
    for clf_name, metrics_df in results.items():
        print(f"\n{clf_name}:")
        print(metrics_df.to_string())
        print(f"\nMean F1-Score: {metrics_df['F1-Score'].mean():.4f}")


def compare_classifiers(results_digits, results_fashion):
    """
    Create comparison summary across classifiers and datasets.
    
    TODO: After running, answer these questions:
    1. Which classifier performed best on MNIST Digits? Why?
    2. Which performed best on MNIST Fashion? Why might it differ?
    3. Which classes were hardest to classify in each dataset?
    4. How does linear SVM compare to kernel SVM? What does this tell you?
    5. Is the added complexity of neural networks justified by performance?
    """
    print(f"\n{'='*80}")
    print("OVERALL COMPARISON")
    print(f"{'='*80}\n")
    
    comparison_data = []
    
    for clf_name in results_digits.keys():
        digits_f1 = results_digits[clf_name]['F1-Score'].mean()
        fashion_f1 = results_fashion[clf_name]['F1-Score'].mean()
        
        comparison_data.append({
            'Classifier': clf_name,
            'Digits F1': f"{digits_f1:.4f}",
            'Fashion F1': f"{fashion_f1:.4f}",
            'Difference': f"{digits_f1 - fashion_f1:.4f}"
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.to_string(index=False))
    
    # TODO: Add your analysis here based on the results


def main():
    """Main execution function."""
    
    # TODO: Update these paths to where your data files are located
    digits_path = 'path/to/mnist_digits.mat'
    fashion_path = 'path/to/mnist_fashion.mat'
    
    # Process MNIST Digits
    print("="*80)
    print("PROCESSING MNIST DIGITS")
    print("="*80)
    xtrain_d, ytrain_d, xtest_d, ytest_d = load_and_preprocess_data(digits_path)
    classifiers_digits = train_classifiers(xtrain_d, ytrain_d, "MNIST Digits")
    results_digits = evaluate_classifiers(classifiers_digits, xtest_d, ytest_d, "MNIST Digits")
    display_results(results_digits, "MNIST Digits")
    
    # Process MNIST Fashion
    print("\n" + "="*80)
    print("PROCESSING MNIST FASHION")
    print("="*80)
    xtrain_f, ytrain_f, xtest_f, ytest_f = load_and_preprocess_data(fashion_path)
    classifiers_fashion = train_classifiers(xtrain_f, ytrain_f, "MNIST Fashion")
    results_fashion = evaluate_classifiers(classifiers_fashion, xtest_f, ytest_f, "MNIST Fashion")
    display_results(results_fashion, "MNIST Fashion")
    
    # Compare results
    compare_classifiers(results_digits, results_fashion)


if __name__ == "__main__":
    main()