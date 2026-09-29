import React from 'react';

export default function VerifierPanel({ result }: { result: any }) {
  const violations = result?.violations || [];
  const checks = [
    ['Inventory Non-negativity', 'INVENTORY'],
    ['Storage Capacity Limits', 'CAPACITY'],
    ['Production Feasibility', 'PRODUCTION'],
    ['Material Conservation', 'CONSERVATION'],
    ['Service Floor (70%)', 'SERVICE'],
    ['Fair Allocation', 'FAIRNESS'],
  ];
  const valid = Boolean(result?.valid);

  return (
    <section className="panel verifier-panel" aria-labelledby="verifier-title" style={{ height: '100%' }}>
      <div className="panel-heading" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '14px' }}>
        <div>
          <h2 id="verifier-title" style={{ fontSize: '1rem', fontWeight: 700, color: '#173b36', margin: 0 }}>
            Safety Review (Verifier Gate)
          </h2>
          <p style={{ fontSize: '0.8rem', color: '#64748b', margin: '2px 0 0 0' }}>
            Deterministic feasibility checks executed before simulator commitment
          </p>
        </div>
        <span style={{
          display: 'inline-flex',
          alignItems: 'center',
          gap: '6px',
          padding: '4px 10px',
          borderRadius: '999px',
          fontSize: '0.75rem',
          fontWeight: 700,
          background: valid ? '#dcfce7' : '#fee2e2',
          color: valid ? '#15803d' : '#b91c1c',
          border: `1px solid ${valid ? '#86efac' : '#fca5a5'}`
        }}>
          {valid ? 'APPROVED' : 'REVISION REQUIRED'}
        </span>
      </div>

      <div style={{
        background: valid ? '#f0fdf4' : '#fef2f2',
        border: `1px solid ${valid ? '#bbf7d0' : '#fecaca'}`,
        borderRadius: '8px',
        padding: '12px 14px',
        marginBottom: '14px',
        display: 'flex',
        alignItems: 'center',
        gap: '12px'
      }}>
        <div style={{
          width: '32px',
          height: '32px',
          borderRadius: '50%',
          background: valid ? '#16a34a' : '#dc2626',
          color: '#fff',
          display: 'grid',
          placeItems: 'center',
          fontSize: '1rem',
          fontWeight: 700,
          flexShrink: 0
        }}>
          {valid ? '✓' : '!'}
        </div>
        <div>
          <strong style={{ fontSize: '0.9rem', color: valid ? '#14532d' : '#7f1d1d', display: 'block' }}>
            {valid ? 'Plan is Safe to Execute' : 'Plan Infeasible — Revisions Enforced'}
          </strong>
          <span style={{ fontSize: '0.75rem', color: valid ? '#166534' : '#991b1b', lineHeight: '1.3' }}>
            {valid
              ? 'All linear operational inequalities and contractual service constraints were satisfied.'
              : 'The proposed agreement violated physical bounds. Verifier required counter-proposals.'}
          </span>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px' }}>
        {checks.map(([label, key]) => {
          const failed = violations.some((v: any) => v.constraint?.toUpperCase().includes(key));
          return (
            <div
              key={key}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
                padding: '6px 10px',
                borderRadius: '6px',
                background: failed ? '#fef2f2' : '#f8fafc',
                border: `1px solid ${failed ? '#fca5a5' : '#e2e8f0'}`,
                fontSize: '0.75rem',
                fontWeight: 600,
                color: failed ? '#b91c1c' : '#334155'
              }}
            >
              <span style={{ color: failed ? '#dc2626' : '#16a34a', fontWeight: 800 }}>
                {failed ? '✕' : '✓'}
              </span>
              <span>{label}</span>
              <span style={{ marginLeft: 'auto', fontSize: '0.7rem', color: failed ? '#dc2626' : '#16a34a' }}>
                {failed ? 'Failed' : 'Passed'}
              </span>
            </div>
          );
        })}
      </div>

      {violations.length > 0 && (
        <div style={{ marginTop: '12px', padding: '10px', background: '#fef2f2', border: '1px solid #fecaca', borderRadius: '6px' }}>
          <strong style={{ fontSize: '0.75rem', color: '#991b1b', display: 'block', marginBottom: '2px' }}>
            Violation Feedback:
          </strong>
          {violations.map((x: any, i: number) => (
            <div key={i} style={{ fontSize: '0.7rem', color: '#b91c1c' }}>
              <strong>[{x.constraint}]</strong> {x.message}
            </div>
          ))}
        </div>
      )}
    </section>
  );
}
