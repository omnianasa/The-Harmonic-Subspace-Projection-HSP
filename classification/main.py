import numpy as np
import time
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from hsp import HSP
from inspector import HSPInspector
from xml.parsers.expat import model
from analyzer import HSPAnalyzer
from hsp_signal import HSPSignalDiagnostics

if __name__ == "__main__":
    print("Fetching MNIST dataset...")
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
    
    #scale to [0, 1]
    X_train, X_test, y_train, y_test = train_test_split(
        X / 255.0, y, test_size=0.15, random_state=42
    )

    model = HSP(ray_dim=512, subspace_dim=15)

    print("Training HSP (Eigen-decomposition phase)...")
    t0 = time.time()
    model.learn(X_train, y_train)
    train_time = time.time() - t0

    print("Inference (Resonance calculation phase)...")
    t1 = time.time()
    preds = model.predict(X_test)
    inference_time = time.time() - t1

    #metrics
    accuracy = np.mean(preds == y_test) * 100
    throughput = len(X_test) / inference_time

    print(f"\n--- HSP Technical Report ---")
    print(f"Train Time:      {train_time:.4f}s")
    print(f"Final Accuracy:   {accuracy:.2f}%")
    print(f"Throughput:      {throughput:.0f} img/sec")

    analyzer = HSPAnalyzer(model)
    analyzer.plot_subspace_overlap()     # Are 4s and 9s overlapping?
    analyzer.plot_spatial_templates()    # What 'shapes' did the model learn?
    analyzer.visualize_latent_rays(X_test[:1000], y_test[:1000]) # t-SNE

    inspector = HSPInspector(model)
    idx = np.random.randint(0, len(X_test))
    inspector.plot_resonance_proof(X_test[idx], y_test[idx])
    inspector.plot_signal_waveform(X_test[idx])

    diag = HSPSignalDiagnostics(model)
    snr_vals = diag.plot_snr_distribution(X_test, y_test)
    print(f"Operational Confidence: {np.mean(snr_vals):.2f} dB")
    diag.plot_phase_plane(X_test[0])