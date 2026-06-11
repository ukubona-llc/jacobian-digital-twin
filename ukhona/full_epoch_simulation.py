import numpy as np
import json
from datetime import datetime
from typing import Dict, Any
from adversarial_perturbation import AdversarialPerturbationGenerator
from jacobian import SimpleJacobian
from event_store import NDJSONEventStore

class DigitalTwinEpochSimulator:
    """
    Full epoch simulation closing the pentad loop:
    Sensation (θ^t) -> Attention -> Intention -> Reaction (Jacobian + error_FGT)
    -> Action (SGD update to L(θ^{t+1}))
    With embodied taste calibration and adversarial training.
    """
    
    def __init__(self, dimension: int = 256, learning_rate: float = 0.01, momentum: float = 0.9):
        self.dimension = dimension
        self.lr = learning_rate
        self.momentum = momentum
        self.event_store = NDJSONEventStore("twin_events.ndjson")
        self.generator = AdversarialPerturbationGenerator(dimension=dimension, event_store=self.event_store)
        self.jacobian = SimpleJacobian(dim=dimension)
        self.velocity = np.zeros(dimension)  # For SGD with momentum
    
    def sgd_update(self, current_state: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        """SGD with momentum for action layer update (L(θ^{t+1}))."""
        self.velocity = self.momentum * self.velocity - self.lr * gradient
        new_state = current_state + self.velocity
        # Embodied clip for realism (taste bounds)
        new_state = np.clip(new_state, -3, 5)
        return new_state
    
    def compute_error_fgt(self, intended_state: np.ndarray, observed_state: np.ndarray) -> np.ndarray:
        """Eigenmode error vs Federated Ground Truth (embodied feedback)."""
        return intended_state - observed_state  # Simplified FGT as target alignment
    
    def run_full_epoch(self, base_state: np.ndarray, num_perturbations: int = 5, 
                      adversary_focus: str = None) -> Dict:
        """Simulate one complete epoch with adversarial training and SGD update."""
        epoch_data = {
            'timestamp': datetime.now().isoformat(),
            'base_state_norm': float(np.linalg.norm(base_state)),
            'perturbations': [],
            'updates': [],
            'final_state_norm': None,
            'kingdomonic_scores': {}
        }
        
        current_state = base_state.copy()
        
        for i in range(num_perturbations):
            adv_type = adversary_focus or np.random.choice(self.generator.adversary_types)
            pert_result = self.generator.generate_perturbation(current_state, adv_type, intensity=0.12 + 0.08 * (i/num_perturbations), targeted=True)
            perturbed = pert_result['perturbed_state']
            pert = pert_result['perturbation']
            
            # Attention: Jacobian sensitivity
            sens = self.jacobian.compute_sensitivity(current_state, pert, layer='attention')
            impact = self.generator.measure_impact(current_state, perturbed, sens['response'])
            
            # Intention: Simple adaptation (perturbation response)
            intended = current_state + sens['response'] * 0.3  # Monumental adaptation
            
            # Reaction: Error FGT
            error = self.compute_error_fgt(intended, perturbed)
            
            # Action: SGD update
            new_state = self.sgd_update(current_state, error)
            
            # Log event
            event = {
                'epoch_step': i,
                'adversary': adv_type,
                'impact': impact,
                'jacobian_metrics': sens['metrics'],
                'error_norm': float(np.linalg.norm(error)),
                'update_norm': float(np.linalg.norm(new_state - current_state))
            }
            self.event_store.append({'type': 'epoch_step', **event})
            
            epoch_data['perturbations'].append(event)
            current_state = new_state
        
        # Final metrics
        epoch_data['final_state_norm'] = float(np.linalg.norm(current_state))
        
        # Kingdomonic scores (compressed 9->5 indicators)
        pedigree_integrity = 1.0 + np.mean([p['impact']['pedigree_loss'] for p in epoch_data['perturbations']])
        adversarial_robustness = 1.0 - np.mean([p['impact'].get('jacobian_sensitivity', 0) for p in epoch_data['perturbations']]) / 2
        embodied_resonance = np.mean([p['impact']['resonance_gain'] for p in epoch_data['perturbations']])
        eigenmode_stability = 1.0 / (1 + np.mean([p['error_norm'] for p in epoch_data['perturbations']]))
        legacy_impact = epoch_data['final_state_norm'] / (epoch_data['base_state_norm'] + 1e-8)
        
        epoch_data['kingdomonic_scores'] = {
            'pedigree_integrity': float(pedigree_integrity),
            'adversarial_robustness': float(adversarial_robustness),
            'embodied_resonance': float(embodied_resonance),
            'eigenmode_stability': float(eigenmode_stability),
            'legacy_impact': float(legacy_impact)
        }
        
        # Overall update log
        self.event_store.append({
            'type': 'full_epoch',
            'epoch_data_summary': {
                'num_steps': num_perturbations,
                'avg_pedigree_loss': float(np.mean([p['impact']['pedigree_loss'] for p in epoch_data['perturbations']])),
                'final_scores': epoch_data['kingdomonic_scores']
            }
        })
        
        return epoch_data

# Test / Demo
if __name__ == "__main__":
    simulator = DigitalTwinEpochSimulator(dimension=256)
    base_state = np.random.randn(256) * 0.5 + 2.0  # Healthy high-pedigree starting point
    
    print("Running full epoch simulation...")
    result = simulator.run_full_epoch(base_state, num_perturbations=4)
    
    print(json.dumps({
        'final_kingdomonic_scores': result['kingdomonic_scores'],
        'final_state_norm': result['final_state_norm']
    }, indent=2))
    
    print("\nRecent events sample:")
    events = simulator.event_store.get_recent(5)
    for e in events[-3:]:
        print(json.dumps({k: v for k, v in e.items() if k in ['type', 'adversary', 'epoch_step']}, indent=2))
