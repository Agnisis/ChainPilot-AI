import { Bar } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';

ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend);

export default function ModelComparisonPanel({ modelRanking, bestModel }) {
  if (!modelRanking || modelRanking.length === 0) return null;

  // Chart Data preparation
  // Sort by rank to ensure the best is first
  const sortedRanking = [...modelRanking].sort((a, b) => a.rank - b.rank);
  const labels = sortedRanking.map(m => m.model);
  const rmseData = sortedRanking.map(m => m.rmse);
  
  // Highlight the best model with a primary color, others with muted color
  const backgroundColors = sortedRanking.map(m => 
    m.model === bestModel ? 'rgba(14, 165, 233, 0.8)' : 'rgba(255, 255, 255, 0.1)'
  );
  const borderColors = sortedRanking.map(m => 
    m.model === bestModel ? 'rgba(14, 165, 233, 1)' : 'rgba(255, 255, 255, 0.3)'
  );

  const data = {
    labels,
    datasets: [
      {
        label: 'RMSE (Lower is Better)',
        data: rmseData,
        backgroundColor: backgroundColors,
        borderColor: borderColors,
        borderWidth: 1,
        borderRadius: 4,
      }
    ]
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: false },
      tooltip: {
        backgroundColor: 'rgba(0, 0, 0, 0.8)',
        titleColor: '#fff',
        bodyColor: '#fff',
      }
    },
    scales: {
      y: {
        grid: { color: 'rgba(255, 255, 255, 0.05)' },
        ticks: { color: 'rgba(255, 255, 255, 0.5)' }
      },
      x: {
        grid: { display: false },
        ticks: { color: 'rgba(255, 255, 255, 0.5)' }
      }
    }
  };

  return (
    <section className="panel model-comparison-panel animate-fade-up delay-2">
      <div className="panel-header">
        <h2>Model Evaluation Leaderboard</h2>
      </div>
      
      <p className="muted" style={{ marginBottom: '24px' }}>
        Comparing error rates and fitness across all trained statistical, machine learning, and deep learning architectures. 
        The winning model ({bestModel}) was automatically selected to generate the forecasts.
      </p>

      <div className="chart-container" style={{ height: '300px', marginBottom: '32px' }}>
        <Bar data={data} options={options} />
      </div>

      <div className="table-responsive">
        <table className="data-table">
          <thead>
            <tr>
              <th>Rank</th>
              <th>Algorithm</th>
              <th>RMSE</th>
              <th>MAE</th>
              <th>R² Score</th>
              <th>Directional Acc.</th>
            </tr>
          </thead>
          <tbody>
            {sortedRanking.map(m => (
              <tr key={m.model} className={m.model === bestModel ? "winner-row" : ""}>
                <td>#{m.rank}</td>
                <td style={{ fontWeight: m.model === bestModel ? '600' : '400', color: m.model === bestModel ? 'var(--primary)' : 'inherit' }}>
                  {m.model} {m.is_ensemble && "🧩"}
                </td>
                <td>{m.rmse.toFixed(2)}</td>
                <td>{m.mae.toFixed(2)}</td>
                <td>{m.r2.toFixed(3)}</td>
                <td>{m.dir_acc ? (m.dir_acc * 100).toFixed(1) + "%" : "—"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}
