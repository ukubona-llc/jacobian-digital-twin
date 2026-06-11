import numpy as np
import json
from datetime import datetime
from typing import Dict, List, Any
from full_epoch_simulation import DigitalTwinEpochSimulator
from event_store import NDJSONEventStore

class MultiEpochDigitalTwinTrainer:
    """
    Multi-epoch adversarial training loop for the digital twin.
    Closes the full pentad: repeated epochs with SGD updates, accumulating legacy.
    Tracks convergence of kingdomonic scores, pedigree stability, and Jacobian evolution.
    """
    
    def __init__(self, dimension: int = 256, num_epochs: int = 20, 
                 learning_rate: float = 0.008, momentum: float = 0.92):
        self.dimension = dimension
        self.num_epochs = num_epochs
        self.simulator = DigitalTwinEpochSimulator(dimension=dimension, 
                                                 learning_rate=learning_rate, 
                                                 momentum=momentum)
        self.event_store = NDJSONEventStore("multi_epoch_training.ndjson")
        self.history = []
        self.base_state = np.random.randn(dimension) * 0.6 + 2.2  # Strong initial pedigree bias
    
    def run_training(self, epochs: int = None, perturbations_per_epoch: int = 5) -> Dict:
        """Run multi-epoch adversarial training loop."""
        if epochs is None:
            epochs = self.num_epochs
            
        training_log = {
            'start_time': datetime.now().isoformat(),
            'dimension': self.dimension,
            'epochs': epochs,
            'perturbations_per_epoch': perturbations_per_epoch,
            'epoch_results': [],
            'final_kingdomonic_trends': {}
        }
        
        current_state = self.base_state.copy()
        
        print(f"Starting {epochs}-epoch adversarial training...")
        
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}...")
            
            # Run full epoch with adversarial perturbations + Jacobian + SGD
            result = self.simulator.run_full_epoch(
                current_state, 
                num_perturbations=perturbations_per_epoch,
                adversary_focus=None  # Random for robustness
            )
            
            current_state = np.array([result['final_state_norm'] / np.sqrt(self.dimension)] * self.dimension)  # Simplified state carry-over
            # In full impl, carry actual updated state from simulator
            
            training_log['epoch_results'].append({
                'epoch': epoch,
                'kingdomonic_scores': result['kingdomonic_scores'],
                'final_state_norm': result['final_state_norm'],
                'avg_pedigree_loss': result['epoch_data_summary'].get('avg_pedigree_loss', 0) if 'epoch_data_summary' in result else 0
            })
            
            # Log to persistent store
            self.event_store.append({
                'type': 'training_epoch',
                'epoch': epoch,
                **result['kingdomonic_scores'],
                'state_norm': result['final_state_norm']
            })
            
            self.history.append(result['kingdomonic_scores'])
        
        # Compute trends
        scores = {k: [e['kingdomonic_scores'][k] for e in training_log['epoch_results']] 
                 for k in training_log['epoch_results'][0]['kingdomonic_scores']}
        
        training_log['final_kingdomonic_trends'] = {
            metric: {
                'final': values[-1],
                'mean': float(np.mean(values)),
                'improvement': values[-1] - values[0] if len(values) > 1 else 0
            } for metric, values in scores.items()
        }
        
        # Overall summary
        self.event_store.append({
            'type': 'training_summary',
            'total_epochs': epochs,
            'final_trends': training_log['final_kingdomonic_trends']
        })
        
        print("Training completed. Legacy impact strengthened.")
        return training_log
    
    def get_training_history(self) -> List[Dict]:
        return self.history

# Demo / Test
if __name__ == "__main__":
    trainer = MultiEpochDigitalTwinTrainer(dimension=128, num_epochs=8, learning_rate=0.007)
    results = trainer.run_training()
    
    print("\n=== Multi-Epoch Training Summary ===")
    print(json.dumps(results['final_kingdomonic_trends'], indent=2))
    
    print("\nRecent training events:")
    events = trainer.event_store.get_recent(5)
    for e in events:
        if e.get('type') in ['training_epoch', 'training_summary']:
            print(json.dumps({k: v for k, v in e.items() if k in ['type', 'epoch', 'pedigree_integrity', 'legacy_impact']}, indent=2))
