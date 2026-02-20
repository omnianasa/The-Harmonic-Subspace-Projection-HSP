import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import time

class HSP:
    """
    The Harmonic Signal Protocol (HSP)
    
    A subspace-based classifier that projects data into a high-dimensional 
    Fourier space and classifies via 'Resonance Energy'—the L2 norm of the 
    projection onto class-specific principal components.
    """
    
    def __init__(self, ray_dim=1024, subspace_dim=10):
        self.ray_dim = ray_dim
        self.subspace_dim = subspace_dim
        self.projection_matrix = None
        self.all_bases = None
        self.global_mean = None
        self.labels = None

    def _init_projection(self, input_dim):
        """Initializes an orthogonal projection matrix for Fourier mapping."""
        rng = np.random.default_rng(42)
        # We use a random normal matrix scaled for the Fourier transform
        W = rng.standard_normal((input_dim, self.ray_dim)) * 0.1
        self.projection_matrix, _ = np.linalg.qr(W)

    def emit_ray(self, x):
        """Maps input vector x into the high-dimensional oscillating feature space."""
        # 1. Linear Projection
        z = np.dot(x, self.projection_matrix)
        
        # 2. Fourier Activation (Sine/Cosine stack)
        # This approximates a RBF kernel in higher dimensions
        activated = np.column_stack([np.cos(z), np.sin(z)])

        # 3. Global Mean Centering (Interference Reduction)
        if self.global_mean is not None:
            activated -= self.global_mean

        # 4. L2 Normalization (Mapping to the Hypersphere)
        norm = np.linalg.norm(activated, axis=1, keepdims=True) + 1e-9
        return activated / norm

    def learn(self, x, y):
        """Learns class-specific subspaces via Eigen-decomposition."""
        if self.projection_matrix is None:
            self._init_projection(x.shape[1])

        # Phase 1: Establish the Global Signal Mean
        initial_rays = self.emit_ray(x)
        self.global_mean = np.mean(initial_rays, axis=0)

        # Phase 2: Compute Class Subspaces
        rays = self.emit_ray(x)
        self.labels = sorted(np.unique(y))
        bases_list = []

        for label in self.labels:
            class_rays = rays[y == label]
            
            # Efficiently find principal components via the covariance matrix
            # Covariance = (X^T * X)
            cov = np.dot(class_rays.T, class_rays)
            _, vh = np.linalg.eigh(cov)

            # Keep the top k eigenvectors as the 'Basis' for this class
            # vh[:, -k:] gives the eigenvectors with the largest eigenvalues
            class_basis = vh[:, -self.subspace_dim:].T
            bases_list.append(class_basis)

        # Stack all bases into one matrix for vectorized prediction
        self.all_bases = np.vstack(bases_list)

    def predict(self, x):
        """Classifies by finding the subspace with the highest Resonance Energy."""
        rays = self.emit_ray(x)
        
        # Project the 'ray' onto every basis vector for every class at once
        # Shape: (n_samples, n_labels * subspace_dim)
        projections = np.dot(rays, self.all_bases.T)
        
        # Reshape to (n_samples, n_classes, subspace_dim)
        energy_tensor = projections.reshape(len(x), len(self.labels), self.subspace_dim)
        
        # Resonance Energy = Sum of squared projections (Squared L2 Norm in subspace)
        scores = np.sum(energy_tensor**2, axis=2)
        
        # Return the label with the maximum resonance
        best_indices = np.argmax(scores, axis=1)
        return np.array([self.labels[i] for i in best_indices])

# ---------------------------------------------------------
# Execution & Verification
# ---------------------------------------------------------

if __name__ == "__main__":
    print("Fetching MNIST dataset...")
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
    
    # Preprocessing: Scale to [0, 1]
    X_train, X_test, y_train, y_test = train_test_split(
        X / 255.0, y, test_size=0.15, random_state=42
    )

    # Initialize and Train
    model = HSP(ray_dim=512, subspace_dim=15)

    print("Training HSP (Eigen-decomposition phase)...")
    t0 = time.time()
    model.learn(X_train, y_train)
    train_time = time.time() - t0

    print("Inference (Resonance calculation phase)...")
    t1 = time.time()
    preds = model.predict(X_test)
    inference_time = time.time() - t1

    # Metrics
    accuracy = np.mean(preds == y_test) * 100
    throughput = len(X_test) / inference_time

    print(f"\n--- HSP Technical Report ---")
    print(f"Train Time:      {train_time:.4f}s")
    print(f"Final Accuracy:   {accuracy:.2f}%")
    print(f"Throughput:      {throughput:.0f} img/sec")