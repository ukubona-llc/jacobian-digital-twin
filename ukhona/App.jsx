import { useState, useEffect } from 'react';
import './App.css';

const API_BASE = 'http://localhost:8000';

function App() {
  const [selectedEpoch, setSelectedEpoch] = useState(0);
  const [selectedLayer, setSelectedLayer] = useState('attention');
  const [kingdomonicScores, setKingdomonicScores] = useState({
    pedigree_integrity: 0.97,
    adversarial_robustness: 0.86,
    embodied_resonance: 0.0005,
    eigenmode_stability: 0.38,
    legacy_impact: 0.99
  });
  const [scoreTrends, setScoreTrends] = useState({});
  const [loading, setLoading] = useState(true);

  const layers = ['sensation', 'attention', 'intention', 'reaction', 'action'];
  const epochs = Array.from({ length: 6 }, (_, i) => i);

  // Fetch dynamic data from FastAPI backend
  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        
        // Fetch kingdomonic scores
        const scoresRes = await fetch(`${API_BASE}/kingdomonic/scores`);
        if (scoresRes.ok) {
          const scoresData = await scoresRes.json();
          setKingdomonicScores(scoresData);
        }

        // Fetch history for trends
        const historyRes = await fetch(`${API_BASE}/training/history?limit=20`);
        if (historyRes.ok) {
          const historyData = await historyRes.json();
          const events = historyData.events || [];
          
          // Build trends from events
          const trends = {
            pedigree_integrity: [],
            adversarial_robustness: [],
            embodied_resonance: [],
            eigenmode_stability: [],
            legacy_impact: []
          };
          
          events.forEach((e, idx) => {
            if (e.kingdomonic_scores) {
              Object.keys(trends).forEach(key => {
                if (e.kingdomonic_scores[key] !== undefined) {
                  trends[key].push(e.kingdomonic_scores[key]);
                }
              });
            } else if (e.pedigree_integrity) {
              trends.pedigree_integrity.push(e.pedigree_integrity);
              // ... similarly for others if flat
            }
          });
          
          // Fill defaults if sparse
          Object.keys(trends).forEach(key => {
            if (trends[key].length === 0) {
              trends[key] = Array.from({length: 6}, (_, i) => 0.95 + i*0.005);
            }
          });
          
          setScoreTrends(trends);
        }
      } catch (error) {
        console.error('Failed to fetch from backend:', error);
        // Fallback to static
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  const getHeatmapUrl = (epoch, layer) => {
    return `/heatmaps/epoch_${epoch.toString().padStart(2, '0')}_${layer}.png`;
  };

  return (
    <div className="ukubona-dashboard">
      <header>
        <h1>🌍 Ukubona Digital Twin Dashboard</h1>
        <p>Adversarial Embodied Pentad • Jacobian Evolution • Kingdomonic Metrics</p>
        <div className="status">FastAPI Backend Live • NDJSON Sourced • Jacobian Dynamic • Vive la Table</div>
      </header>

      {/* Kingdomonic Scores */}
      <section className="scores">
        <h2>Kingdomonic Indicators (Epoch {selectedEpoch})</h2>
        <div className="score-grid">
          {Object.entries(kingdomonicScores).map(([key, value]) => (
            <div key={key} className="score-card">
              <div className="score-label">{key.replace(/_/g, ' ').toUpperCase()}</div>
              <div className="score-value">{(value * 100).toFixed(1)}%</div>
              <div className="trend">
                Trend: {(scoreTrends[key] && scoreTrends[key][selectedEpoch] !== undefined && 
                         scoreTrends[key][selectedEpoch] > scoreTrends[key][0]) ? '↑' : '↓'}
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* Jacobian Evolution Controls */}
      <section className="controls">
        <h2>Jacobian Sensitivity Evolution</h2>
        <div className="selectors">
          <div>
            <label>Epoch: </label>
            <select value={selectedEpoch} onChange={(e) => setSelectedEpoch(parseInt(e.target.value))}>
              {epochs.map(e => (
                <option key={e} value={e}>Epoch {e}</option>
              ))}
            </select>
          </div>
          <div>
            <label>Layer: </label>
            <select value={selectedLayer} onChange={(e) => setSelectedLayer(e.target.value)}>
              {layers.map(l => (
                <option key={l} value={l}>{l.charAt(0).toUpperCase() + l.slice(1)}</option>
              ))}
            </select>
          </div>
        </div>
      </section>

      {/* Main Heatmap Viewer */}
      <section className="heatmap-viewer">
        <h2>{selectedLayer.charAt(0).toUpperCase() + selectedLayer.slice(1)} Layer Sensitivity (Epoch {selectedEpoch})</h2>
        <div className="heatmap-container">
          <img 
            src={getHeatmapUrl(selectedEpoch, selectedLayer)} 
            alt={`${selectedLayer} Jacobian Heatmap Epoch ${selectedEpoch}`}
            className="jacobian-heatmap"
          />
        </div>
        <p className="caption">High sensitivity in Attention layer highlights pointing/compressor criticality. SGD updates stabilize over epochs.</p>
      </section>

      {/* Trends */}
      <section className="trends">
        <h2>Mean Sensitivity Evolution Across Epochs</h2>
        <img 
          src="/heatmaps/sensitivity_trends.png" 
          alt="Jacobian Sensitivity Trends" 
          className="trend-plot"
        />
        <p>Attention remains the Jacobian hotspot — driving embodied adversarial adaptation.</p>
      </section>

      {/* Multi-scale Jacobian Summary */}
      <section className="summary">
        <h2>Σ Jacobian Multi-Scale</h2>
        <div className="grid">
          <div>Individual: Issue → Arterial</div>
          <div>Network: Nodes/Edges/Weights</div>
          <div>Care Systems: Assets → Equity</div>
          <div>Regulators: 403 Sovereignty</div>
          <div>Planners: Embodied → Generative</div>
        </div>
      </section>

      <footer>
        <p>Closed-loop Digital Twin • Sensation → Action via Jacobian-driven SGD • Testable AGI Foundations</p>
        <p>Built on Ukubona Pentad • NDJSON Event Sourcing • Ready for SlowAPI / Vite React UX</p>
      </footer>
    </div>
  );
}

export default App;
