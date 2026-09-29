import React, { useState } from 'react';

type Forecast = {
  status?: string;
  recommendation?: any;
  alternatives?: any[];
  horizon_days?: number;
  note?: string;
  message?: string;
  runtime_tuning?: any;
};

export default function RecoveryForecast({ forecast }: { forecast: Forecast | undefined }) {
  const [showTable, setShowTable] = useState(true);

  if (!forecast) return null;
  if (forecast.status !== 'ready') {
    return (
      <section className="panel forecast-panel forecast-unavailable" aria-labelledby="forecast-title">
        <div>
          <h2 id="forecast-title" style={{ fontSize: '1rem', color: '#173b36' }}>ARDN Advisory Recovery Forecast</h2>
          <p style={{ fontSize: '0.85rem', color: '#64748b' }}>The neural predictor is unavailable for this run.</p>
        </div>
        <small>{forecast.message}</small>
      </section>
    );
  }

  const r = forecast.recommendation;
  const options = forecast.alternatives || [];

  return (
    <section className="panel forecast-panel" aria-labelledby="forecast-title">
      <div className="panel-heading" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '14px' }}>
        <div>
          <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
            <h2 id="forecast-title" style={{ fontSize: '1rem', fontWeight: 700, color: '#173b36', margin: 0 }}>
              ARDN Advisory Consequence Projection
            </h2>
            <span style={{ fontSize: '0.7rem', padding: '2px 8px', borderRadius: '4px', background: '#e0f2fe', color: '#0369a1', fontWeight: 700, border: '1px solid #bae6fd' }}>
              Neural Rollout (15 Days) · Advisory Only
            </span>
          </div>
          <p style={{ fontSize: '0.8rem', color: '#64748b', margin: '3px 0 0 0' }}>
            Action-conditioned graph neural network predictions evaluated before execution commitment
          </p>
        </div>

        <button
          onClick={() => setShowTable(!showTable)}
          style={{
            background: '#f1f5f9',
            border: '1px solid #cbd5e1',
            borderRadius: '6px',
            padding: '4px 10px',
            fontSize: '0.75rem',
            fontWeight: 600,
            color: '#334155'
          }}
        >
          {showTable ? 'Hide 6-Option Table' : 'Compare All 6 Interventions'}
        </button>
      </div>

      {/* Recommended Action Card */}
      <div style={{
        background: '#f8fafc',
        border: '1px solid #cbd5e1',
        borderRadius: '8px',
        padding: '14px 16px',
        marginBottom: '14px',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        flexWrap: 'wrap',
        gap: '16px'
      }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '4px' }}>
            <span style={{ fontSize: '0.7rem', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.05em', color: '#0284c7' }}>
              Recommended Candidate Action
            </span>
            <span style={{ fontSize: '0.7rem', padding: '1px 6px', borderRadius: '3px', background: '#dcfce7', color: '#15803d', fontWeight: 700 }}>
              ★ Decision Score: {r.ranking_score?.toFixed(3)}
            </span>
          </div>
          <strong style={{ fontSize: '1.15rem', color: '#0f172a' }}>{r.label}</strong>
          <p style={{ margin: '4px 0 0 0', fontSize: '0.8rem', color: '#475569', maxWidth: '620px', lineHeight: '1.4' }}>
            {r.rationale || 'Selected via multi-attribute decision score balancing recovery time, service loss, and cost.'}
          </p>
        </div>

        <div style={{
          background: '#eff6ff',
          border: '1px solid #bfdbfe',
          borderRadius: '8px',
          padding: '10px 16px',
          textAlign: 'right'
        }}>
          <div style={{ fontSize: '1.4rem', fontWeight: 800, color: '#1d4ed8' }}>
            {r.recovery_days} <span style={{ fontSize: '0.85rem', fontWeight: 500, color: '#64748b' }}>± {r.recovery_uncertainty_days} d</span>
          </div>
          <span style={{ fontSize: '0.7rem', color: '#64748b', fontWeight: 600 }}>Predicted Recovery Time (90% target)</span>
        </div>
      </div>

      {/* 4 Metric Chips */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '10px', marginBottom: '14px' }}>
        <div style={{ background: '#f8fafc', border: '1px solid #e2e8f0', borderRadius: '6px', padding: '10px 12px' }}>
          <span style={{ fontSize: '0.7rem', fontWeight: 600, color: '#64748b', display: 'block' }}>Predicted Incremental Cost</span>
          <div style={{ fontSize: '1.1rem', fontWeight: 700, color: '#0f172a', margin: '2px 0' }}>${r.predicted_cost}</div>
          <span style={{ fontSize: '0.7rem', color: '#64748b' }}>Interval: ${r.cost_interval?.[0]}–${r.cost_interval?.[1]}</span>
        </div>

        <div style={{ background: '#f8fafc', border: '1px solid #e2e8f0', borderRadius: '6px', padding: '10px 12px' }}>
          <span style={{ fontSize: '0.7rem', fontWeight: 600, color: '#64748b', display: 'block' }}>Predicted Service Loss</span>
          <div style={{ fontSize: '1.1rem', fontWeight: 700, color: '#0f172a', margin: '2px 0' }}>{Number(r.service_loss).toFixed(3)}</div>
          <span style={{ fontSize: '0.7rem', color: '#64748b' }}>Drop during shock window</span>
        </div>

        <div style={{ background: '#f8fafc', border: '1px solid #e2e8f0', borderRadius: '6px', padding: '10px 12px' }}>
          <span style={{ fontSize: '0.7rem', fontWeight: 600, color: '#64748b', display: 'block' }}>Severe Overflow Risk</span>
          <div style={{ fontSize: '1.1rem', fontWeight: 700, color: '#0284c7', margin: '2px 0' }}>{r.risk_label?.split(' ')[0] || '< 1%'}</div>
          <span style={{ fontSize: '0.7rem', color: '#64748b' }}>Backlog &gt; 1.5× capacity</span>
        </div>

        <div style={{ background: '#f8fafc', border: '1px solid #e2e8f0', borderRadius: '6px', padding: '10px 12px' }}>
          <span style={{ fontSize: '0.7rem', fontWeight: 600, color: '#64748b', display: 'block' }}>Novelty / OOD Score (D_M)</span>
          <div style={{ fontSize: '1.1rem', fontWeight: 700, color: r.ood_score > 15 ? '#d97706' : '#15803d', margin: '2px 0' }}>
            {r.ood_score}
          </div>
          <span style={{ fontSize: '0.7rem', color: '#64748b' }}>{r.ood_status} (Threshold: 15.0)</span>
        </div>
      </div>

      {/* Comparison table */}
      {showTable && (
        <div style={{ border: '1px solid #e2e8f0', borderRadius: '6px', overflow: 'hidden' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.75rem', textAlign: 'left' }}>
            <thead>
              <tr style={{ background: '#f1f5f9', borderBottom: '1px solid #cbd5e1' }}>
                <th style={{ padding: '8px 10px', color: '#475569', fontWeight: 700 }}>Candidate Action</th>
                <th style={{ padding: '8px 10px', color: '#475569', fontWeight: 700 }}>Recovery</th>
                <th style={{ padding: '8px 10px', color: '#475569', fontWeight: 700 }}>Uncertainty</th>
                <th style={{ padding: '8px 10px', color: '#475569', fontWeight: 700 }}>Incremental Cost</th>
                <th style={{ padding: '8px 10px', color: '#475569', fontWeight: 700 }}>Service Loss</th>
                <th style={{ padding: '8px 10px', color: '#475569', fontWeight: 700 }}>Risk</th>
                <th style={{ padding: '8px 10px', color: '#475569', fontWeight: 700 }}>Novelty (D_M)</th>
                <th style={{ padding: '8px 10px', color: '#475569', fontWeight: 700 }}>Feasibility</th>
                <th style={{ padding: '8px 10px', color: '#475569', fontWeight: 700 }}>Score</th>
              </tr>
            </thead>
            <tbody>
              {options.map((opt: any) => {
                const isSelected = opt.action === r.action;
                return (
                  <tr
                    key={opt.action}
                    style={{
                      background: isSelected ? '#f0fdf4' : '#fff',
                      borderBottom: '1px solid #e2e8f0',
                      fontWeight: isSelected ? 600 : 'normal'
                    }}
                  >
                    <td style={{ padding: '8px 10px', color: '#0f172a' }}>
                      {opt.label} {isSelected && <span style={{ color: '#16a34a', fontSize: '0.65rem', fontWeight: 700 }}>★ Selected</span>}
                    </td>
                    <td style={{ padding: '8px 10px', color: '#0f172a' }}>{opt.recovery_days} d</td>
                    <td style={{ padding: '8px 10px', color: '#64748b' }}>± {opt.recovery_uncertainty_days} d</td>
                    <td style={{ padding: '8px 10px', color: '#0f172a' }}>${opt.predicted_cost}</td>
                    <td style={{ padding: '8px 10px', color: '#0f172a' }}>{Number(opt.service_loss).toFixed(3)}</td>
                    <td style={{ padding: '8px 10px', color: '#64748b' }}>{opt.risk_label?.split(' ')[0] || '< 1%'}</td>
                    <td style={{ padding: '8px 10px', color: opt.ood_score > 15 ? '#d97706' : '#15803d' }}>
                      {opt.ood_score}
                    </td>
                    <td style={{ padding: '8px 10px' }}>
                      <span style={{
                        padding: '2px 6px',
                        borderRadius: '3px',
                        fontSize: '0.65rem',
                        fontWeight: 700,
                        background: opt.feasibility === 'Feasible' ? '#dcfce7' : '#fef3c7',
                        color: opt.feasibility === 'Feasible' ? '#15803d' : '#b45309',
                      }}>
                        {opt.feasibility}
                      </span>
                    </td>
                    <td style={{ padding: '8px 10px', color: '#0284c7', fontWeight: 700 }}>
                      {opt.ranking_score?.toFixed(3) || '—'}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}
