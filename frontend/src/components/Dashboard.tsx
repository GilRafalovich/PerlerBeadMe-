import React, { useState, useEffect } from 'react';
import './Dashboard.css';

interface MetricsSummary {
  total_evaluated: number;
  average_iou_percent: number;
  average_chamfer_distance: number;
}

interface RunResult {
  image: string;
  category: string;
  iou: number;
  chamfer_distance: number;
  llm_score?: number | null;
  llm_critique?: string | null;
}

interface VerificationData {
  summary: MetricsSummary;
  runs: RunResult[];
}

export const Dashboard: React.FC = () => {
  const [data, setData] = useState<VerificationData | null>(null);
  const [loading, setLoading] = useState(false);
  const [triggering, setTriggering] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchMetrics = async () => {
    setLoading(true);
    try {
      const response = await fetch('http://localhost:8000/api/metrics');
      const json = await response.json();
      if (!response.ok) {
        setError(json.message || json.detail || "Error fetching metrics: " + response.statusText);
        setData(null);
      } else if (json.status === 'error') {
        setError(json.message);
        setData(null);
      } else if (!json.summary) {
        setError("Invalid metrics data format received.");
        setData(null);
      } else {
        setData(json);
        setError(null);
      }
    } catch (err) {
      console.error(err);
      setError("Failed to fetch metrics from server. Ensure backend is running.");
      setData(null);
    } finally {
      setLoading(false);
    }
  };

  const handleRunVerification = async () => {
    setTriggering(true);
    try {
      const response = await fetch('http://localhost:8000/api/run_verification?limit=100', {
        method: 'POST'
      });
      const result = await response.json();
      if (result.status === 'success') {
        alert("Background verification started! Refresh the dashboard in a few minutes to see updated results.");
      } else {
        alert("Failed to start verification: " + result.message);
      }
    } catch (err) {
      console.error(err);
      alert("Error triggering verification pipeline.");
    } finally {
      setTriggering(false);
    }
  };

  useEffect(() => {
    fetchMetrics();
  }, []);

  return (
    <div className="dashboard-container">
      <div className="dashboard-header">
        <h2>Pipeline Verification Metrics</h2>
        <div className="dashboard-actions">
          <button className="btn" onClick={fetchMetrics} disabled={loading}>
            {loading ? 'Refreshing...' : 'Refresh Metrics'}
          </button>
          <button className="btn btn-primary" onClick={handleRunVerification} disabled={triggering}>
            {triggering ? 'Starting...' : 'Run 100 Samples'}
          </button>
        </div>
      </div>

      {error && <div className="alert-error">{error}</div>}

      {data && (
        <>
          <div className="metrics-cards">
            <div className="card glass-panel">
              <h3>Samples Evaluated</h3>
              <div className="metric-value">{data.summary.total_evaluated}</div>
            </div>
            <div className="card glass-panel">
              <h3>Avg 3D IoU</h3>
              <div className="metric-value">
                {data.summary.average_iou_percent.toFixed(2)}%
              </div>
              <p className="metric-note">Expect low due to intentional hollowing</p>
            </div>
            <div className="card glass-panel">
              <h3>Avg Chamfer Distance</h3>
              <div className="metric-value">
                {data.summary.average_chamfer_distance.toFixed(4)}
              </div>
              <p className="metric-note">Lower is better for structural match</p>
            </div>
          </div>

          <div className="glass-panel" style={{ marginTop: '2rem' }}>
            <h3>Individual Run Details</h3>
            <div className="table-container">
              <table className="metrics-table">
                <thead>
                  <tr>
                    <th>Image</th>
                    <th>Category</th>
                    <th>IoU (%)</th>
                    <th>Chamfer Distance</th>
                    <th>LLM Score</th>
                    <th>LLM Critique</th>
                  </tr>
                </thead>
                <tbody>
                  {data.runs.map((run, idx) => (
                    <tr key={idx}>
                      <td>{run.image}</td>
                      <td>{run.category}</td>
                      <td>{(run.iou * 100).toFixed(2)}%</td>
                      <td>{run.chamfer_distance.toFixed(4)}</td>
                      <td>{run.llm_score ? `${run.llm_score}/10` : 'N/A'}</td>
                      <td>{run.llm_critique || 'N/A'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}
    </div>
  );
};
