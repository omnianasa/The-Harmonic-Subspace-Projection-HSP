import matplotlib.pyplot as plt
import numpy as np

class HSPInspector:
    
    def __init__(self, model):
        self.model = model

    def plot_resonance_proof(self, x_single, true_label):
        ray = self.model.emit_ray(x_single.reshape(1, -1))
        n_classes = len(self.model.labels)
        
        #energy
        all_projs = np.dot(ray, self.model.all_bases.T).flatten()
        energy_spectrum = all_projs**2
        class_energies = energy_spectrum.reshape(n_classes, self.model.subspace_dim).sum(axis=1)
        predicted = self.model.labels[np.argmax(class_energies)]

        fig = plt.figure(figsize=(15, 6))
        
        #The Input
        ax1 = plt.subplot2grid((2, 3), (0, 0))
        ax1.imshow(x_single.reshape(28, 28), cmap='gray')
        ax1.set_title(f"Input (True: {true_label})")
        ax1.axis('off')

        #The Energy Spectrum 
        ax2 = plt.subplot2grid((2, 3), (0, 1), colspan=2)
        ax2.plot(energy_spectrum, color='darkorange', drawstyle='steps-mid')
        ax2.fill_between(range(len(energy_spectrum)), energy_spectrum, color='orange', alpha=0.3)
        for i in range(n_classes + 1):
            ax2.axvline(i * self.model.subspace_dim, color='black', lw=1, ls='--')
        ax2.set_title("Resonance Spectrum (Per-Basis Energy)")

        #inal Decision Bar
        ax3 = plt.subplot2grid((2, 3), (1, 0), colspan=3)
        colors = ['red' if l == predicted else 'skyblue' for l in self.model.labels]
        ax3.bar(self.model.labels, class_energies, color=colors)
        ax3.set_title(f"Final Decision: {predicted}")
        ax3.set_ylabel("Total Resonance")
        
        plt.tight_layout()
        plt.show()

    def plot_signal_waveform(self, x_single):
        ray = self.model.emit_ray(x_single.reshape(1, -1)).flatten()
        
        plt.figure(figsize=(15, 3))
        plt.plot(ray, color='royalblue', lw=0.5)
        plt.title("High-Dimensional Ray Waveform (Signal Signature)")
        plt.xlabel("Ray Dimension")
        plt.ylabel("Amplitude")
        plt.xlim(0, len(ray))
        plt.show()