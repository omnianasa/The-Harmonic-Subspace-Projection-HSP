import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
import seaborn as sns
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Model Imports
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB

# Set Scientific Seeds
np.random.seed(42)

class HSP:
    """
    The Harmonic Subspace Projection (HSP)

    This model performs classification by mapping input data into a high dimensional
    Fourier feature space (Rays) and then learning class specific subspaces via
    Eigen decomposition.

    Decision making is based on the 'Resonance Energy'
    between the input ray and learned class bases.

    Attributes:
        ray_dim (int): Dimensionality of the random projection.
        subspace_dim (int): Number of principal components kept for each class.
    """

    def __init__(self, ray_dim=1024, subspace_dim=10):
        self.ray_dim = ray_dim
        self.subspace_dim = subspace_dim
        self.projection_matrix = None
        self.all_bases = None
        self.global_mean = None
        self.labels = None

    def _init_projection(self, input_dim):
        """Initializes an orthogonal projection matrix using QR decomposition."""
        rng = np.random.default_rng(42)
        W = rng.standard_normal((input_dim, self.ray_dim)) * 0.1
        self.projection_matrix, _ = np.linalg.qr(W)

    def emit_ray(self, x):
        """Maps input vector x into the high-dimensional oscillating feature space."""
        z = np.dot(x, self.projection_matrix)

        # Apply Sine/Cosine transformation (Fourier Features)
        activated = np.column_stack([np.cos(z), np.sin(z)])

        # Apply Global Mean Centering for interference reduction
        if self.global_mean is not None:
            activated -= self.global_mean

        # L2 Normalization to place rays on a hypersphere
        norm = np.linalg.norm(activated, axis=1, keepdims=True) + 1e-9
        return activated / norm

    def learn(self, x, y):
        """Trains class subspaces by identifying principal components in ray-space."""
        if self.projection_matrix is None:
            self._init_projection(x.shape[1])

        # Step 1: Calculate Global Mean across the training set
        initial_rays = self.emit_ray(x)
        self.global_mean = np.mean(initial_rays, axis=0)

        # Step 2: Extract Class-specific Subspaces
        rays = self.emit_ray(x)
        labels = np.unique(y)
        bases_list = []

        for label in sorted(labels):
            class_rays = rays[y == label]

            # Use covariance-based Eigen-decomposition for subspace estimation
            cov = np.dot(class_rays.T, class_rays)
            _, vh = np.linalg.eigh(cov)

            # Extract the top eigenvectors (the 'Basis' of the class)
            bases_list.append(vh[:, -self.subspace_dim:].T)

        self.all_bases = np.vstack(bases_list)
        self.labels = sorted(labels)

    def predict(self, x):
        """Predicts class label by maximizing projected energy across subspaces."""
        rays = self.emit_ray(x)

        # Project image ray onto all class bases simultaneously
        projections = np.dot(rays, self.all_bases.T)

        # Calculate Squared Energy per class
        energy = projections.reshape(len(x), len(self.labels), self.subspace_dim)**2
        scores = np.sum(energy, axis=2)

        return np.array([self.labels[i] for i in np.argmax(scores, axis=1)])

# --- Benchmark Execution ---

X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
X_train, X_test, y_train, y_test = train_test_split(X/255.0, y, test_size=5000, train_size=20000, random_state=42)

models = {
    "HSP": HSP(512, 10),
    "Logistic Reg": LogisticRegression(max_iter=100),
    "Ridge Clf": RidgeClassifier(),
    "KNN (k=3)": KNeighborsClassifier(n_neighbors=3),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(n_estimators=50),
    "Extra Trees": ExtraTreesClassifier(n_estimators=50),
    "SVM (RBF)": SVC(kernel='rbf'),
    "MLP (128,64)": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=20),
    "Naive Bayes": GaussianNB()
}

results_list = []

for name, clf in models.items():
    print(f"Testing: {name}")
    t0 = time.time()
    if name == "HSP": clf.learn(X_train, y_train)
    else: clf.fit(X_train, y_train)
    t_train = time.time() - t0

    t1 = time.time()
    preds = clf.predict(X_test)
    t_inf = time.time() - t1

    acc = accuracy_score(y_test, preds)
    results_list.append({
        "Model": name,
        "Accuracy": acc,
        "Train Time (s)": t_train,
        "Inference Time (s)": t_inf,
        "Throughput (img/s)": len(X_test)/t_inf,
        "Scaling": "Linear" if name in ["Turbo HSP", "Ridge Clf", "Naive Bayes"] else "Non-Linear",
        "Interpretability": "High" if name in ["Turbo HSP", "Decision Tree", "Logistic Reg"] else "Low"
    })

df = pd.DataFrame(results_list)
df.to_csv("model_comparison_results.csv", index=False)