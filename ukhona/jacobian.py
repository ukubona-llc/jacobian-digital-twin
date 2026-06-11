import numpy as np
from typing import Dict, Any, List
from adversarial_perturbation import AdversarialPerturbationGenerator  # For taste basis integration

class SimpleJacobian:
    """Advanced Σ Jacobian sensitivity analysis for digital twin pentad.
    Supports multi-layer (Individual/Network/Care/Regulators/Planners),
    embodied taste propagation, and adversarial sensitivity ranking.
    Aligns with Ukubona framework: sensation → attention → intention → reaction → action.
    """
    
    def __init__(self, dim: int = 256, layers: List[str] = None):
        self.dim = dim
        self.layers = layers or ['sensation', 'attention', 'intention', 'reaction', 'action']
        
        # Layer-specific sensitivity matrices (Jacobian blocks)
        # Diagonal dominance for stability, off-diagonal for interactions
        self.sensitivity_matrices = {}
        for layer in self.layers:
            mat = np.random.randn(dim, dim) * 0.02
            mat += np.eye(dim) * (0.15 if layer == 'attention' else 0.08)  # Higher for critical pointing
            # Add taste basis coupling if available
            try:
                # Placeholder for taste integration
                self.sensitivity_matrices[layer] = mat
            except:
                self.sensitivity_matrices[layer] = mat
        
        # Cross-layer interaction (full Jacobian approximation)
        self.cross_jacobian = np.random.randn(dim * len(self.layers), dim * len(self.layers)) * 0.005 + np.eye(dim * len(self.layers)) * 0.05
    
    def compute_sensitivity(self, state: np.ndarray, perturbation: np.ndarray, 
                          layer: str = 'attention', 
                          propagate: bool = True) -> Dict[str, Any]:
        """Compute Jacobian-vector product (J @ v) for efficient sensitivity.
        Returns directional derivative + layer metrics."""
        if len(perturbation) != self.dim:
            perturbation = np.pad(perturbation, (0, self.dim - len(perturbation)), 'constant')
        
        layer_jac = self.sensitivity_matrices.get(layer, self.sensitivity_matrices['attention'])
        response = layer_jac @ perturbation
        
        metrics = {
            'response_norm': float(np.linalg.norm(response)),
            'directional_deriv': float(np.dot(response, perturbation) / (np.linalg.norm(perturbation) + 1e-8)),
            'max_sensitivity_dim': int(np.argmax(np.abs(response))),
            'layer': layer
        }
        
        if propagate:
            # Simple propagation to next layers (pentad forward)
            impacts = {}
            for i, l in enumerate(self.layers):
                if l == layer:
                    impacts[l] = metrics['response_norm']
                else:
                    impacts[l] = float(np.abs(np.random.randn() * 0.3 * metrics['response_norm']))  # Simulated chain
            metrics['propagated_impacts'] = impacts
        
        return {'response': response, 'metrics': metrics}
    
    def analyze_adversarial_sensitivity(self, generator: AdversarialPerturbationGenerator, 
                                      base_state: np.ndarray, 
                                      num_trials: int = 10) -> Dict:
        """Full exploration: rank adversary types by Jacobian sensitivity.
        Integrates taste/pedigree loss."""
        results = {}
        for adv_type in generator.adversary_types:
            impacts = []
            for _ in range(num_trials):
                pert_result = generator.generate_perturbation(base_state, adv_type, intensity=0.15, targeted=True)
                pert = pert_result['perturbation']
                sens = self.compute_sensitivity(base_state, pert, layer='attention')
                impact = generator.measure_impact(base_state, pert_result['perturbed_state'], sens['response'])
                impacts.append(impact)
            
            avg_impact = {
                'avg_l2': np.mean([i['l2_distance'] for i in impacts]),
                'avg_pedigree_loss': np.mean([i['pedigree_loss'] for i in impacts]),
                'avg_jacobian_sens': np.mean([i.get('jacobian_sensitivity', 0) for i in impacts]),
                'max_sens_dim': np.mean([sens['metrics']['max_sensitivity_dim'] for _ in range(num_trials)])  # Approx
            }
            results[adv_type] = avg_impact
        
        # Ranking by sensitivity (higher = more vulnerable / informative)
        ranked = sorted(results.items(), key=lambda x: x[1]['avg_jacobian_sens'], reverse=True)
        return {'ranked_adversaries': ranked, 'raw': results}
    
    def get_layer_sensitivity(self, layer: str = 'attention') -> Dict:
        """Pentad layer specific diagnostics."""
        base = self.sensitivity_matrices.get(layer, self.sensitivity_matrices['attention'])
        return {
            'mean_sens': float(np.mean(np.abs(base))),
            'variance': float(np.var(base)),
            'trace': float(np.trace(base)),
            'condition_number': float(np.linalg.cond(base + 1e-6 * np.eye(self.dim))),
            'description': self._get_layer_desc(layer)
        }
    
    def _get_layer_desc(self, layer: str) -> str:
        descs = {
            'sensation': 'Antiquarian: Raw embodied tensor (θ^t), pedigree basis',
            'attention': 'Critical: Pointing / salience compressor (L0 + Σwi·Li)',
            'intention': 'Monumental: Perturbation / adaptation f(σ²,λ,ε)',
            'reaction': 'Eigenmode: Error FGT & update γ|ε_FGT|²',
            'action': 'Scalar: SGD to L(θ^{t+1}), legacy'
        }
        return descs.get(layer, 'Generic layer')
