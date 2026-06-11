import numpy as np
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
import random
from event_store import NDJSONEventStore

class AdversarialPerturbationGenerator:
    """
    Generates adversarial perturbations for the digital twin's closed-loop system.
    Focuses on embodied sensation (θ^t), attention (Jacobian sensitivity), 
    intention perturbation, etc. Inspired by Ukubona pentad and taste calibration.
    """
    
    def __init__(self, dimension: int = 512, seed: int = 42, event_store: Optional[NDJSONEventStore] = None):
        self.dimension = dimension
        np.random.seed(seed)
        random.seed(seed)
        self.perturbation_history: List[Dict] = []
        self.event_store = event_store or NDJSONEventStore("adversarial_events.ndjson")
        # Basis vectors for embodied taste (from calibration: Esther, Sandra, Tura-like)
        self.taste_basis = {
            'pedigree': np.random.randn(dimension) * 0.1 + 1.0,  # High fidelity
            'bland': np.random.randn(dimension) * 0.05,           # Collapse risk
            'resonant': np.random.randn(dimension) * 0.15         # Embodied strength
        }
        self.adversary_types = ['cultural_drift', 'supply_shock', 'sensory_noise', 'intentional_mispoint', 'pedigree_loss']
    
    def generate_perturbation(self, 
                            base_state: np.ndarray, 
                            adversary_type: str = None, 
                            intensity: float = 0.1,
                            targeted: bool = False) -> Dict[str, Any]:
        """Generate perturbation at sensation level or specific layer."""
        if adversary_type is None:
            adversary_type = random.choice(self.adversary_types)
        
        if len(base_state) != self.dimension:
            base_state = np.zeros(self.dimension)  # Fallback
        
        perturbation = np.zeros_like(base_state)
        
        if adversary_type == 'cultural_drift':
            # Gradual shift in meaning/taste vectors
            perturbation = np.random.randn(self.dimension) * intensity * 0.5
            # Bias towards bland
            perturbation += self.taste_basis['bland'] * intensity
        
        elif adversary_type == 'supply_shock':
            # Sudden drop in key dimensions (resources/attention)
            shock_dims = np.random.choice(self.dimension, int(self.dimension * 0.2), replace=False)
            perturbation[shock_dims] = -np.abs(np.random.randn(len(shock_dims))) * intensity * 2
        
        elif adversary_type == 'sensory_noise':
            # Embodied sensor noise (Gaussian + salt-pepper)
            perturbation = np.random.normal(0, intensity, self.dimension)
            # Salt-pepper for realism
            mask = np.random.random(self.dimension) < 0.05
            perturbation[mask] = np.random.choice([-intensity*5, intensity*5], size=np.sum(mask))
        
        elif adversary_type == 'intentional_mispoint':
            # Misalign attention/salience (compressor matrix attack)
            perturbation = np.random.randn(self.dimension) * intensity
            # Target specific "pointing" directions
            if targeted:
                perturbation[:self.dimension//4] *= -2  # Disrupt critical attention
        
        elif adversary_type == 'pedigree_loss':
            # Direct attack on taste/pedigree integrity (403 risk)
            perturbation = -self.taste_basis['pedigree'] * intensity * 1.5
            # Add noise to prevent easy detection
            perturbation += np.random.randn(self.dimension) * intensity * 0.3
        
        # Normalize and scale
        norm = np.linalg.norm(perturbation)
        if norm > 0:
            perturbation = perturbation / norm * intensity * np.sqrt(self.dimension)
        
        perturbed_state = base_state + perturbation
        
        # Clip to reasonable bounds (embodied realism)
        perturbed_state = np.clip(perturbed_state, -5, 5)
        
        event = {
            'timestamp': datetime.now().isoformat(),
            'adversary_type': adversary_type,
            'intensity': intensity,
            'perturbation_norm': float(np.linalg.norm(perturbation)),
            'targeted': targeted,
            'base_norm': float(np.linalg.norm(base_state)),
            'perturbed_norm': float(np.linalg.norm(perturbed_state))
        }
        
        self.perturbation_history.append(event)
        
        # Log to append-only NDJSON for SlowAPI / Jacobian
        self.event_store.append({
            'type': 'adversarial_perturbation',
            'adversary_type': adversary_type,
            'intensity': intensity,
            'perturbation_norm': float(np.linalg.norm(perturbation)),
            'impact_preview': {'l2': float(np.linalg.norm(perturbed_state - base_state))}
        })
        
        return {
            'perturbed_state': perturbed_state,
            'perturbation': perturbation,
            'metadata': event
        }
    
    def measure_impact(self, original_state: np.ndarray, perturbed_state: np.ndarray, 
                      jacobian_sensitivity: Optional[np.ndarray] = None) -> Dict:
        """Measure embodied feedback: sensation delta, Jacobian response."""
        delta = perturbed_state - original_state
        impact = {
            'l2_distance': float(np.linalg.norm(delta)),
            'cosine_similarity': float(np.dot(original_state, perturbed_state) / 
                                     (np.linalg.norm(original_state) * np.linalg.norm(perturbed_state) + 1e-8)),
            'pedigree_loss': float(np.dot(delta, self.taste_basis['pedigree']) / self.dimension),
            'resonance_gain': float(np.dot(delta, self.taste_basis['resonant']) / self.dimension)
        }
        
        if jacobian_sensitivity is not None:
            # Sensitivity along perturbation direction
            sensitivity = np.abs(np.dot(jacobian_sensitivity.flatten(), delta) / (np.linalg.norm(delta) + 1e-8))
            impact['jacobian_sensitivity'] = float(sensitivity)
        
        return impact
    
    def get_history(self) -> List[Dict]:
        return self.perturbation_history
    
    def simulate_adversarial_epoch(self, base_state: np.ndarray, num_perturbations: int = 5) -> List[Dict]:
        """Run a batch for training loop closure."""
        results = []
        for _ in range(num_perturbations):
            adv_type = random.choice(self.adversary_types)
            pert = self.generate_perturbation(base_state, adv_type, intensity=random.uniform(0.05, 0.25))
            impact = self.measure_impact(base_state, pert['perturbed_state'])
            results.append({
                'adversary': adv_type,
                'perturbation': pert['metadata'],
                'impact': impact
            })
        return results

# Example usage / test
if __name__ == "__main__":
    from jacobian import SimpleJacobian
    generator = AdversarialPerturbationGenerator(dimension=256)
    base = np.random.randn(256) * 0.5 + 1.0  # Healthy embodied state (high pedigree)
    jacobian = SimpleJacobian(256)
    
    results = generator.simulate_adversarial_epoch(base, 3)
    for res in results:
        pert = generator.generate_perturbation(base, res['adversary'], intensity=0.15, targeted=True)
        impact = generator.measure_impact(base, pert['perturbed_state'], jacobian.compute_sensitivity(base, pert['perturbation']))
        print(json.dumps({'adversary': res['adversary'], 'impact': impact}, indent=2))
    
    print("\nEvent store sample:")
    print(generator.event_store.read_all()[-2:])
