import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import KFold
from scipy.io import loadmat
import warnings
warnings.filterwarnings(‘ignore’)

# =============================================================================

# PART 0: Load and Visualize the Data

# =============================================================================

# Load the data from cs.mat

data = loadmat(‘cs.mat’)
y = data[‘y’].flatten()  # Measurements (1300,)
A = data[‘A’]            # Measurement matrix (1300, 2500)
x_true = data[‘x’].flatten()  # True image (2500,)

print(“Data dimensions:”)
print(f”  y (measurements): {y.shape}”)
print(f”  A (sensing matrix): {A.shape}”)
print(f”  x_true (true image): {x_true.shape}”)
print(f”\nSparsity of true image: {np.sum(x_true != 0)}/2500 non-zero pixels”)

# Visualize the true image (50x50)

plt.figure(figsize=(6, 5))
plt.imshow(x_true.reshape(50, 50), cmap=‘gray’)
plt.title(‘True Sparse Image’)
plt.colorbar()
plt.savefig(‘true_image.png’, dpi=150, bbox_inches=‘tight’)
plt.show()

# =============================================================================

# PART 1: Lasso Regression with 10-Fold Cross-Validation

# =============================================================================

print(”\n” + “=”*70)
print(“PART 1: LASSO REGRESSION”)
print(”=”*70)

# Define range of lambda values to test (logarithmic spacing)

# For Lasso, sklearn uses alpha = lambda / (2*n), so we need to adjust

n_samples = len(y)
lambda_values = np.logspace(-2, 3, 50)  # Test 50 lambda values
alpha_values = lambda_values / (2 * n_samples)  # Convert to sklearn’s alpha

# Setup 10-fold cross-validation

kf = KFold(n_splits=10, shuffle=True, random_state=42)

# Store cross-validation errors for each lambda

cv_errors_lasso = []

print(”\nPerforming 10-fold cross-validation for Lasso…”)

for alpha in alpha_values:
fold_errors = []

```
for train_idx, val_idx in kf.split(y):
    # Split data
    y_train, y_val = y[train_idx], y[val_idx]
    A_train, A_val = A[train_idx, :], A[val_idx, :]
    
    # Fit Lasso on training data
    lasso = Lasso(alpha=alpha, max_iter=10000, tol=1e-4)
    lasso.fit(A_train, y_train)
    
    # Compute validation error
    y_pred = A_val @ lasso.coef_
    mse = np.mean((y_val - y_pred) ** 2)
    fold_errors.append(mse)

# Average error across all folds
cv_errors_lasso.append(np.mean(fold_errors))
```

cv_errors_lasso = np.array(cv_errors_lasso)

# Find optimal lambda

optimal_idx_lasso = np.argmin(cv_errors_lasso)
optimal_lambda_lasso = lambda_values[optimal_idx_lasso]
optimal_alpha_lasso = alpha_values[optimal_idx_lasso]

print(f”\nOptimal lambda (Lasso): {optimal_lambda_lasso:.4f}”)
print(f”Corresponding alpha: {optimal_alpha_lasso:.6f}”)
print(f”Minimum CV error: {cv_errors_lasso[optimal_idx_lasso]:.6f}”)

# Fit final Lasso model with optimal lambda on full dataset

lasso_final = Lasso(alpha=optimal_alpha_lasso, max_iter=10000, tol=1e-4)
lasso_final.fit(A, y)
x_lasso = lasso_final.coef_

print(f”\nRecovered image sparsity: {np.sum(np.abs(x_lasso) > 1e-6)}/2500 non-zero pixels”)
print(f”Reconstruction error: ||x_true - x_lasso||_2 = {np.linalg.norm(x_true - x_lasso):.4f}”)

# =============================================================================

# PART 2: Ridge Regression with 10-Fold Cross-Validation

# =============================================================================

print(”\n” + “=”*70)
print(“PART 2: RIDGE REGRESSION”)
print(”=”*70)

# For Ridge, sklearn uses alpha = lambda, so no conversion needed

lambda_values_ridge = np.logspace(-2, 3, 50)

cv_errors_ridge = []

print(”\nPerforming 10-fold cross-validation for Ridge…”)

for lam in lambda_values_ridge:
fold_errors = []

```
for train_idx, val_idx in kf.split(y):
    # Split data
    y_train, y_val = y[train_idx], y[val_idx]
    A_train, A_val = A[train_idx, :], A[val_idx, :]
    
    # Fit Ridge on training data
    ridge = Ridge(alpha=lam, max_iter=10000)
    ridge.fit(A_train, y_train)
    
    # Compute validation error
    y_pred = A_val @ ridge.coef_
    mse = np.mean((y_val - y_pred) ** 2)
    fold_errors.append(mse)

cv_errors_ridge.append(np.mean(fold_errors))
```

cv_errors_ridge = np.array(cv_errors_ridge)

# Find optimal lambda

optimal_idx_ridge = np.argmin(cv_errors_ridge)
optimal_lambda_ridge = lambda_values_ridge[optimal_idx_ridge]

print(f”\nOptimal lambda (Ridge): {optimal_lambda_ridge:.4f}”)
print(f”Minimum CV error: {cv_errors_ridge[optimal_idx_ridge]:.6f}”)

# Fit final Ridge model with optimal lambda on full dataset

ridge_final = Ridge(alpha=optimal_lambda_ridge, max_iter=10000)
ridge_final.fit(A, y)
x_ridge = ridge_final.coef_

print(f”\nRecovered image sparsity: {np.sum(np.abs(x_ridge) > 1e-6)}/2500 non-zero pixels”)
print(f”Reconstruction error: ||x_true - x_ridge||_2 = {np.linalg.norm(x_true - x_ridge):.4f}”)

# =============================================================================

# VISUALIZATION: Cross-Validation Error Curves

# =============================================================================

plt.figure(figsize=(12, 5))

# Lasso CV curve

plt.subplot(1, 2, 1)
plt.semilogx(lambda_values, cv_errors_lasso, ‘b-’, linewidth=2)
plt.axvline(optimal_lambda_lasso, color=‘r’, linestyle=’–’,
label=f’Optimal λ = {optimal_lambda_lasso:.4f}’)
plt.xlabel(‘λ (regularization parameter)’, fontsize=12)
plt.ylabel(‘Cross-Validation Error (MSE)’, fontsize=12)
plt.title(‘Lasso: 10-Fold Cross-Validation Error’, fontsize=14, fontweight=‘bold’)
plt.grid(True, alpha=0.3)
plt.legend()

# Ridge CV curve

plt.subplot(1, 2, 2)
plt.semilogx(lambda_values_ridge, cv_errors_ridge, ‘g-’, linewidth=2)
plt.axvline(optimal_lambda_ridge, color=‘r’, linestyle=’–’,
label=f’Optimal λ = {optimal_lambda_ridge:.4f}’)
plt.xlabel(‘λ (regularization parameter)’, fontsize=12)
plt.ylabel(‘Cross-Validation Error (MSE)’, fontsize=12)
plt.title(‘Ridge: 10-Fold Cross-Validation Error’, fontsize=14, fontweight=‘bold’)
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.savefig(‘cv_error_curves.png’, dpi=150, bbox_inches=‘tight’)
plt.show()

# =============================================================================

# VISUALIZATION: Recovered Images Comparison

# =============================================================================

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# True image

im1 = axes[0].imshow(x_true.reshape(50, 50), cmap=‘gray’, vmin=0, vmax=1)
axes[0].set_title(‘True Image\n(416 non-zero pixels)’, fontsize=12, fontweight=‘bold’)
axes[0].axis(‘off’)
plt.colorbar(im1, ax=axes[0], fraction=0.046)

# Lasso recovered image

im2 = axes[1].imshow(x_lasso.reshape(50, 50), cmap=‘gray’, vmin=0, vmax=1)
axes[1].set_title(f’Lasso Recovery\n(λ={optimal_lambda_lasso:.4f})\n’ +
f’Error: {np.linalg.norm(x_true - x_lasso):.4f}’,
fontsize=12, fontweight=‘bold’)
axes[1].axis(‘off’)
plt.colorbar(im2, ax=axes[1], fraction=0.046)

# Ridge recovered image

im3 = axes[2].imshow(x_ridge.reshape(50, 50), cmap=‘gray’, vmin=0, vmax=1)
axes[2].set_title(f’Ridge Recovery\n(λ={optimal_lambda_ridge:.4f})\n’ +
f’Error: {np.linalg.norm(x_true - x_ridge):.4f}’,
fontsize=12, fontweight=‘bold’)
axes[2].axis(‘off’)
plt.colorbar(im3, ax=axes[2], fraction=0.046)

plt.tight_layout()
plt.savefig(‘image_comparison.png’, dpi=150, bbox_inches=‘tight’)
plt.show()

# =============================================================================

# ADDITIONAL ANALYSIS: Sparsity Pattern

# =============================================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Coefficient magnitude comparison

sorted_true = np.sort(np.abs(x_true))[::-1]
sorted_lasso = np.sort(np.abs(x_lasso))[::-1]
sorted_ridge = np.sort(np.abs(x_ridge))[::-1]

axes[0].semilogy(range(500), sorted_true[:500], ‘k-’, linewidth=2, label=‘True’, alpha=0.7)
axes[0].semilogy(range(500), sorted_lasso[:500], ‘b-’, linewidth=2, label=‘Lasso’, alpha=0.7)
axes[0].semilogy(range(500), sorted_ridge[:500], ‘g-’, linewidth=2, label=‘Ridge’, alpha=0.7)
axes[0].set_xlabel(‘Coefficient Rank’, fontsize=12)
axes[0].set_ylabel(‘Absolute Coefficient Value’, fontsize=12)
axes[0].set_title(‘Sorted Coefficient Magnitudes (Top 500)’, fontsize=14, fontweight=‘bold’)
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

# Histogram of non-zero coefficients

axes[1].hist(x_true[x_true != 0], bins=30, alpha=0.5, label=‘True’, color=‘black’, edgecolor=‘black’)
axes[1].hist(x_lasso[np.abs(x_lasso) > 1e-6], bins=30, alpha=0.5, label=‘Lasso’, color=‘blue’, edgecolor=‘blue’)
axes[1].hist(x_ridge[np.abs(x_ridge) > 1e-6], bins=30, alpha=0.5, label=‘Ridge’, color=‘green’, edgecolor=‘green’)
axes[1].set_xlabel(‘Coefficient Value’, fontsize=12)
axes[1].set_ylabel(‘Frequency’, fontsize=12)
axes[1].set_title(‘Distribution of Non-Zero Coefficients’, fontsize=14, fontweight=‘bold’)
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(‘sparsity_analysis.png’, dpi=150, bbox_inches=‘tight’)
plt.show()

# =============================================================================

# SUMMARY COMPARISON

# =============================================================================

print(”\n” + “=”*70)
print(“SUMMARY: LASSO vs RIDGE COMPARISON”)
print(”=”*70)

print(”\n{:<30} {:<20} {:<20}”.format(“Metric”, “Lasso”, “Ridge”))
print(”-” * 70)
print(”{:<30} {:<20.4f} {:<20.4f}”.format(
“Optimal λ”, optimal_lambda_lasso, optimal_lambda_ridge))
print(”{:<30} {:<20.6f} {:<20.6f}”.format(
“Min CV Error”, cv_errors_lasso[optimal_idx_lasso], cv_errors_ridge[optimal_idx_ridge]))
print(”{:<30} {:<20} {:<20}”.format(
“Non-zero coefficients”,
np.sum(np.abs(x_lasso) > 1e-6),
np.sum(np.abs(x_ridge) > 1e-6)))
print(”{:<30} {:<20.4f} {:<20.4f}”.format(
“Reconstruction Error”,
np.linalg.norm(x_true - x_lasso),
np.linalg.norm(x_true - x_ridge)))
print(”{:<30} {:<20.4f} {:<20.4f}”.format(
“Relative Error (%)”,
100 * np.linalg.norm(x_true - x_lasso) / np.linalg.norm(x_true),
100 * np.linalg.norm(x_true - x_ridge) / np.linalg.norm(x_true)))

print(”\n” + “=”*70)
print(“CONCLUSION”)
print(”=”*70)
print(”\nWhich approach gives a better recovered image?”)
print(f”\nBased on the reconstruction error:”)
if np.linalg.norm(x_true - x_lasso) < np.linalg.norm(x_true - x_ridge):
print(”  → LASSO provides superior recovery!”)
print(f”    Lasso error: {np.linalg.norm(x_true - x_lasso):.4f}”)
print(f”    Ridge error: {np.linalg.norm(x_true - x_ridge):.4f}”)
print(”\n  This is expected because:”)
print(”  • The true image is SPARSE (only 416/2500 non-zero pixels)”)
print(”  • Lasso’s L1 penalty induces sparsity by setting coefficients exactly to 0”)
print(”  • Ridge’s L2 penalty only shrinks coefficients, creating a ‘blurry’ solution”)
else:
print(”  → RIDGE provides superior recovery (unexpected!)”)
print(”  This would suggest the problem structure favors smoothness over sparsity”)

print(”\n” + “=”*70)