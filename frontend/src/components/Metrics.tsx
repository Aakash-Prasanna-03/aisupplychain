import React from 'react';

export default function Metrics({ state }: { state: any }) {
  const h = state?.history || [];
  const avg = h.length ? h.reduce((a: number, x: any) => a + x.service_level, 0) / h.length : 1.0;
  const interventions = state?.negotiation?.filter((x: any) => x.speaker === 'Verifier' && x.valid === false).length || 0;
  const horizon = state?.experiment?.simulation_horizon || 12;
  const currentDay = state?.day || 0;
  const disrDuration = state?.disruption?.duration || 5;

  return (
    <section className="panel metrics-panel" aria-labelledby="metrics-title" style={{ height: '100%' }}>
      <div className="panel-heading" style={{ marginBottom: '14px' }}>
        <div>
          <h2 id="metrics-title" style={{ fontSize: '1rem', fontWeight: 700, color: '#173b36', margin: 0 }}>
            Scenario Pulse
          </h2>
          <p style={{ fontSize: '0.8rem', color: '#64748b', margin: '2px 0 0 0' }}>
            Ground-truth simulator performance after verified recovery execution
          </p>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
        <div style={{ background: '#f8fafc', padding: '12px', borderRadius: '8px', border: '1px solid #e2e8f0' }}>
          <span style={{ fontSize: '0.75rem', fontWeight: 600, color: '#64748b', display: 'block' }}>
            Simulation Horizon
          </span>
          <div style={{ fontSize: '1.4rem', fontWeight: 800, color: '#0f172a', margin: '2px 0' }}>
            Day {currentDay} <small style={{ fontSize: '0.75rem', color: '#64748b', fontWeight: 500 }}>of {horizon}</small>
          </div>
          <span style={{ fontSize: '0.7rem', color: '#0284c7' }}>
            Disruption active days 1–{disrDuration}
          </span>
        </div>

        <div style={{ background: '#f8fafc', padding: '12px', borderRadius: '8px', border: '1px solid #e2e8f0' }}>
          <span style={{ fontSize: '0.75rem', fontWeight: 600, color: '#64748b', display: 'block' }}>
            Mean Customer Service
          </span>
          <div style={{ fontSize: '1.4rem', fontWeight: 800, color: '#15803d', margin: '2px 0' }}>
            {(avg * 100).toFixed(1)}%
          </div>
          <span style={{ fontSize: '0.7rem', color: '#15803d' }}>
            Protected by buffer stock
          </span>
        </div>

        <div style={{ background: '#f8fafc', padding: '12px', borderRadius: '8px', border: '1px solid #e2e8f0' }}>
          <span style={{ fontSize: '0.75rem', fontWeight: 600, color: '#64748b', display: 'block' }}>
            Total Simulation Cost
          </span>
          <div style={{ fontSize: '1.4rem', fontWeight: 800, color: '#0f172a', margin: '2px 0' }}>
            ${Number(state?.total_cost || 0).toFixed(2)}
          </div>
          <span style={{ fontSize: '0.7rem', color: '#64748b' }}>
            Holding + penalty costs
          </span>
        </div>

        <div style={{ background: '#f8fafc', padding: '12px', borderRadius: '8px', border: '1px solid #e2e8f0' }}>
          <span style={{ fontSize: '0.75rem', fontWeight: 600, color: '#64748b', display: 'block' }}>
            Verifier Interventions
          </span>
          <div style={{ fontSize: '1.4rem', fontWeight: 800, color: '#b45309', margin: '2px 0' }}>
            {interventions} <small style={{ fontSize: '0.75rem', color: '#64748b', fontWeight: 500 }}>rounds</small>
          </div>
          <span style={{ fontSize: '0.7rem', color: '#b45309' }}>
            Infeasible proposals blocked
          </span>
        </div>
      </div>
    </section>
  );
}
