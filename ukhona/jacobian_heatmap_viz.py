import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
from typing import Dict, List, Any
from jacobian import SimpleJacobian
from multi_epoch_training import MultiEpochDigitalTwinTrainer
from adversarial_perturbation import AdversarialPerturbationGenerator

class JacobianEvolutionVisualizer:
    """Generates heatmaps showing Jacobian sensitivity evolution across epochs/layers."""
    
    def __init__(self, dimension: int = 128):
        self.dimension = dimension
        self.jacobian = SimpleJacobian(dim=dimension)
        self.layers = ['sensation', 'attention', 'intention', 'reaction', 'action']
    
    def extract_jacobian_snapshots(self, num_epochs: int = 8) -> List[np.ndarray]:
        """Simulate evolution by perturbing Jacobian matrices slightly each 'epoch'."""
        snapshots = []
        for epoch in range(num_epochs):
            # Evolve: slight random walk on sensitivities (mimics training updates)
            evolved = {}
            for layer in self.layers:
                mat = self.jacobian.sensitivity_matrices[layer].copy()
                # Simulate SGD-like update to Jacobian (sensitivity adaptation)
                update = np.random.randn(self.dimension, self.dimension) * (0.005 * (1 + epoch/num_epochs))
                mat += update
                # Normalize for stability
                mat = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-8)
                evolved[layer] = mat
            snapshots.append(evolved)
        return snapshots
    
    def plot_layer_heatmap(self, matrix: np.ndarray, layer: str, epoch: int, save_path: str = None):
        """Single layer heatmap."""
        plt.figure(figsize=(8, 6))
        plt.imshow(np.abs(matrix[:50, :50]), cmap='viridis', aspect='auto')  # Submatrix for visibility
        plt.colorbar(label='|Sensitivity|')
        plt.title(f'Jacobian Sensitivity Heatmap - {layer.capitalize()} (Epoch {epoch})')
        plt.xlabel('Target Dimensions')
        plt.ylabel('Source Dimensions')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
    def plot_evolution_heatmaps(self, num_epochs: int = 8, output_dir: str = "jacobian_heatmaps"):
        """Generate full evolution visualization suite."""
        Path(output_dir).mkdir(exist_ok=True)
        snapshots = self.extract_jacobian_snapshots(num_epochs)
        
        # Key metrics over epochs
        mean_sens = {layer: [] for layer in self.layers}
        
        for epoch, snap in enumerate(snapshots):
            for layer in self.layers:
                mat = snap[layer]
                mean_sens[layer].append(np.mean(np.abs(mat)))
                
                # Save individual heatmaps (subsample for clarity)
                save_path = f"{output_dir}/epoch_{epoch:02d}_{layer}.png"
                self.plot_layer_heatmap(mat, layer, epoch, save_path)
        
        # Summary trend plot
        plt.figure(figsize=(10, 6))
        for layer, values in mean_sens.items():
            plt.plot(range(num_epochs), values, marker='o', label=layer.capitalize())
        plt.title('Jacobian Mean Sensitivity Evolution Across Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Mean Absolute Sensitivity')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{output_dir}/sensitivity_trends.png")
        plt.close()
        
        print(f"✅ Heatmaps generated in {output_dir}/")
        print("   - Per-epoch layer heatmaps")
        print("   - Sensitivity trends plot")
        return output_dir
    
    def run_demo(self):
        """Run full visualization."""
        print("Generating Jacobian evolution heatmaps...")
        out_dir = self.plot_evolution_heatmaps(num_epochs=6)
        # List files
        files = list(Path(out_dir).glob("*.png"))
        print(f"Generated {len(files)} visualization files.")
        for f in sorted(files)[:5]:  # Sample
            print(f"  - {f.name}")

if __name__ == "__main__":
    viz = JacobianEvolutionVisualizer(dimension=128)
    viz.run_demo()
