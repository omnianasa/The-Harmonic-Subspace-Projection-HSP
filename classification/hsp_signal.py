import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class HSPSignalDiagnostics:
    """
    Evaluates the signal quality of the HSP system using 
    communication theory metrics like SNR and Phase distribution.
    """
    
    def __init__(self, model):
        self.model = model

    def calculate_snr(self, X, y):
        """
        Calculates the Signal-to-Noise Ratio for each prediction.
        Signal = Energy in the correct class subspace.
        Noise = Average energy across all incorrect subspaces.
        """
    
        rays = self.model.emit_ray(X)
        projections = np.dot(rays, self.model.all_bases.T)

        energy_cube = projections.reshape(
            len(X), len(self.model.labels), self.model.subspace_dim
        )**2

        scores = np.sum(energy_cube, axis=2) 
        
        snr_list = []
        for i in range(len(y)):

            true_idx = self.model.labels.index(y[i])
            
            signal = scores[i, true_idx]
            noise_mask = np.ones(len(self.model.labels), dtype=bool)
            noise_mask[true_idx] = False
            noise = np.mean(scores[i, noise_mask])

            snr_db = 10 * np.log10(signal / (noise + 1e-9))
            snr_list.append(snr_db)
            
        return np.array(snr_list)

    def plot_snr_distribution(self, X, y):
        snr_values = self.calculate_snr(X, y)
        mean_snr = np.mean(snr_values)

        plt.figure(figsize=(10, 6))
        sns.histplot(snr_values, kde=True, color='forestgreen', bins=30)
        plt.axvline(mean_snr, color='red', linestyle='--', 
                    label=f'Mean System SNR: {mean_snr:.2f} dB')
        
        plt.title("HSP Signal Clarity: SNR Distribution", fontsize=14)
        plt.xlabel("Clarity (Decibels - dB)")
        plt.ylabel("Frequency (Number of Samples)")
        plt.legend()
        plt.grid(alpha=0.2)
        plt.show()
        
        return snr_values

    def plot_phase_plane(self, x_sample):
        """
        Visualizes the Sine/Cosine distribution of a single ray projection.
        Ideally, these should form a circular distribution on the unit disc.
        """

        z = np.dot(x_sample.reshape(1, -1), self.model.projection_matrix)
        
        cos_part = np.cos(z).flatten()
        sin_part = np.sin(z).flatten()
        
        plt.figure(figsize=(6, 6))
        plt.scatter(cos_part, sin_part, alpha=0.4, s=15, color='purple', edgecolors='white', lw=0.5)
        circle = plt.Circle((0, 0), 1, color='black', fill=False, linestyle='--', alpha=0.3)
        plt.gca().add_patch(circle)
        
        plt.title("HSP White Box: Phase Plane Balance", fontsize=14)
        plt.xlabel("Cosine Component (In-Phase)")
        plt.ylabel("Sine Component (Quadrature)")
        plt.axhline(0, color='black', lw=1, alpha=0.5)
        plt.axvline(0, color='black', lw=1, alpha=0.5)
        plt.grid(True, alpha=0.2)
        plt.axis('equal')
        plt.show()