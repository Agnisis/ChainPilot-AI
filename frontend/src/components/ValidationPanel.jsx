export default function ValidationPanel({ analysis }) {
  const { cv_scores, tuning_results, feature_importance, statistical_tests, best_model } = analysis;

  if (!cv_scores && !tuning_results && !feature_importance && !statistical_tests) {
    return null; // Nothing to show if fast mode was on or no advanced features used
  }

  return (
    <section className="panel advanced-ml-panel animate-fade-up delay-3">
      <h2>Advanced Data Science Validation</h2>
      <p className="subtitle">Model evaluation, interpretability, and tuning metrics.</p>

      <div className="ml-grid">
        {/* SHAP Feature Importance */}
        {feature_importance && feature_importance.length > 0 && (
          <div className="ml-card">
            <h3>SHAP Feature Importance</h3>
            <p className="note">Top drivers for {best_model} predictions.</p>
            <div className="table-responsive">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Feature</th>
                    <th>Importance (Mean |SHAP|)</th>
                    <th>Direction</th>
                  </tr>
                </thead>
                <tbody>
                  {feature_importance.map((f, i) => (
                    <tr key={i}>
                      <td>{f.feature}</td>
                      <td>{f.importance.toFixed(4)}</td>
                      <td className={f.direction.includes("Positive") ? "text-success" : f.direction.includes("Negative") ? "text-danger" : ""}>
                        {f.direction}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* Statistical Tests */}
        {statistical_tests && statistical_tests.length > 0 && (
          <div className="ml-card">
            <h3>Statistical Diagnostics</h3>
            <p className="note">Ensuring data stationarity and residual white noise.</p>
            <div className="table-responsive">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Test</th>
                    <th>p-value</th>
                    <th>Result</th>
                  </tr>
                </thead>
                <tbody>
                  {statistical_tests.map((test, i) => (
                    <tr key={i}>
                      <td>{test.test_name}</td>
                      <td>{test.p_value.toFixed(4)}</td>
                      <td>
                        <span className="badge">{test.result}</span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* Walk-Forward CV */}
        {cv_scores && cv_scores[best_model] && (
          <div className="ml-card">
            <h3>Walk-Forward Cross Validation</h3>
            <p className="note">Time-series aware rolling window evaluation ({best_model}).</p>
            <div className="table-responsive">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Fold</th>
                    <th>Train Size</th>
                    <th>Test Size</th>
                    <th>RMSE</th>
                    <th>MAPE</th>
                  </tr>
                </thead>
                <tbody>
                  {cv_scores[best_model].map((fold, i) => (
                    <tr key={i}>
                      <td>Fold {fold.fold}</td>
                      <td>{fold.train_size}</td>
                      <td>{fold.test_size}</td>
                      <td>{fold.metrics.RMSE.toFixed(2)}</td>
                      <td>{fold.metrics.MAPE.toFixed(2)}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* Optuna Tuning */}
        {tuning_results && tuning_results.length > 0 && (
          <div className="ml-card">
            <h3>Optuna Hyperparameter Optimization</h3>
            <p className="note">Bayesian search over {tuning_results[0]?.n_trials} trials.</p>
            <div className="table-responsive">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Model</th>
                    <th>Best RMSE</th>
                    <th>Parameters</th>
                  </tr>
                </thead>
                <tbody>
                  {tuning_results.map((res, i) => (
                    <tr key={i}>
                      <td>{res.model}</td>
                      <td>{res.best_score.toFixed(2)}</td>
                      <td className="code-cell">
                        {Object.entries(res.best_params)
                          .map(([k, v]) => `${k}=${typeof v === 'number' ? v.toFixed(3).replace(/\.?0+$/, '') : v}`)
                          .join(", ")}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>
    </section>
  );
}
