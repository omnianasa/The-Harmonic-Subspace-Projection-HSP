import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import numpy as np

class HSPAnalyzer:
    """Tools for analyzing the global structure and separation of HSP subspaces."""
    
    def __init__(self, model):
        self.model = model

    def plot_subspace_overlap(self):
        """Visualizes how similar class subspaces are to each other."""
        n_classes = len(self.model.labels)
        similarity = np.zeros((n_classes, n_classes))
        
        # Reshape: (n_classes, subspace_dim, ray_feature_dim)
        bases = self.model.all_bases.reshape(n_classes, self.model.subspace_dim, -1)
        
        for i in range(n_classes):
            for j in range(n_classes):
                # Alignment: Sum of squared projections of one subspace onto another
                proj = np.dot(bases[i], bases[j].T)
                similarity[i, j] = np.sum(proj**2) / self.model.subspace_dim

        plt.figure(figsize=(10, 8))
        sns.heatmap(similarity, annot=True, xticklabels=self.model.labels, 
                    yticklabels=self.model.labels, cmap='viridis', fmt=".2f")
        plt.title("Subspace Similarity (Overlap) Map")
        plt.show()

    def visualize_latent_rays(self, X_sample, y_sample):
        """Reduces high-dim rays to 2D to see class clustering."""
        rays = self.model.emit_ray(X_sample)
        tsne = TSNE(n_components=2, perplexity=30, random_state=42)
        rays_2d = tsne.fit_transform(rays)
        
        plt.figure(figsize=(10, 7))
        for label in np.unique(y_sample):
            mask = y_sample == label
            plt.scatter(rays_2d[mask, 0], rays_2d[mask, 1], label=f"Digit {label}", s=10, alpha=0.6)
        
        plt.legend()
        plt.title("t-SNE Visualization of High-Dimensional Rays")
        plt.show()

    def plot_spatial_templates(self):
        """Maps the high-dim subspace back to the 28x28 pixel grid."""
        n_classes = len(self.model.labels)
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        plt.suptitle("White Box: Learned Spatial Templates", fontsize=16)

        bases_reshaped = self.model.all_bases.reshape(n_classes, self.model.subspace_dim, -1)

        for i in range(n_classes):
            # Focus on the 'cosine' part of the Fourier feature for spatial mapping
            class_basis_part = bases_reshaped[i][:, :self.model.ray_dim] 

            # Map weights back to input space via the transpose of the projection
            importance = np.dot(np.abs(class_basis_part), np.abs(self.model.projection_matrix.T))
            mean_importance = np.mean(importance, axis=0).reshape(28, 28)

            ax = axes[i//5, i%5]
            ax.imshow(mean_importance, cmap='hot')
            ax.set_title(f"Class: {self.model.labels[i]}")
            ax.axis('off')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()