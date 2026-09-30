import React from 'react';
import type { Node } from '../types';

export default function Network({ nodes, isDisruptionActive = false, networkStatus }: { nodes: Node[]; isDisruptionActive?: boolean; networkStatus?: string }) {
  const anyDisrupted = nodes.some(n => n.status === 'DISRUPTED' || n.status === 'CONSTRAINED');
  const anyBuffering = nodes.some(n => n.status === 'BUFFERING');
  const anyRecovering = nodes.some(n => n.status === 'RECOVERING');

  let networkStatusText = networkStatus || 'Operating Normally';
  let badgeBg = '#dcfce7';
  let badgeColor = '#15803d';
  let badgeBorder = '#86efac';

  if (isDisruptionActive) {
    networkStatusText = networkStatus || (anyDisrupted ? 'Active Disruption' : 'Active Disruption (Buffer Protected)');
    badgeBg = anyDisrupted ? '#fee2e2' : '#e0f2fe';
    badgeColor = anyDisrupted ? '#b91c1c' : '#0369a1';
    badgeBorder = anyDisrupted ? '#fca5a5' : '#7dd3fc';
  } else if (anyDisrupted) {
    networkStatusText = 'Active Upstream Disruption';
    badgeBg = '#fee2e2';
    badgeColor = '#b91c1c';
    badgeBorder = '#fca5a5';
  } else if (anyBuffering) {
    networkStatusText = 'Buffer Protected (Shock Absorbed)';
    badgeBg = '#e0f2fe';
    badgeColor = '#0369a1';
    badgeBorder = '#7dd3fc';
  } else if (anyRecovering) {
    networkStatusText = 'Recovering (Stabilizing Buffers)';
    badgeBg = '#f3e8ff';
    badgeColor = '#7e22ce';
    badgeBorder = '#d8b4fe';
  }

  const getStatusBadge = (status: string) => {
    switch (status?.toUpperCase()) {
      case 'DISRUPTED':
        return { bg: '#fee2e2', color: '#b91c1c', border: '#fca5a5', label: 'Disrupted' };
      case 'CONSTRAINED':
        return { bg: '#fef3c7', color: '#b45309', border: '#fcd34d', label: 'Constrained' };
      case 'BUFFERING':
        return { bg: '#e0f2fe', color: '#0369a1', border: '#7dd3fc', label: 'Buffering' };
      case 'RECOVERING':
        return { bg: '#f3e8ff', color: '#7e22ce', border: '#d8b4fe', label: 'Recovering' };
      case 'NORMAL':
      default:
        return { bg: '#dcfce7', color: '#15803d', border: '#86efac', label: 'Normal' };
    }
  };

  return (
    <section className="panel network-panel" aria-labelledby="network-title" style={{ height: '100%' }}>
      <div className="panel-heading" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
        <div>
          <h2 id="network-title" style={{ fontSize: '1rem', fontWeight: 700, color: '#173b36', margin: 0 }}>
            Network Physical Flow
          </h2>
          <p style={{ fontSize: '0.8rem', color: '#64748b', margin: '2px 0 0 0' }}>
            Facility inventories, capacities, and operational states across supply chain tiers
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
          background: badgeBg,
          color: badgeColor,
          border: `1px solid ${badgeBorder}`
        }}>
          <span style={{ width: '6px', height: '6px', borderRadius: '50%', background: badgeColor }} />
          {networkStatusText}
        </span>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '10px' }}>
        {nodes.map((n, i) => {
          const s = getStatusBadge(n.status);
          const pct = Math.min(100, Math.round((n.inventory / Math.max(n.capacity, 1)) * 100));

          return (
            <article
              key={n.id}
              style={{
                background: '#f8fafc',
                border: '1px solid #e2e8f0',
                borderRadius: '8px',
                padding: '12px',
                display: 'flex',
                flexDirection: 'column',
                justifyContent: 'space-between',
                position: 'relative'
              }}
            >
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '4px' }}>
                  <span style={{ fontSize: '0.65rem', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.05em', color: '#64748b' }}>
                    {n.id}
                  </span>
                  <span style={{
                    fontSize: '0.65rem',
                    fontWeight: 700,
                    padding: '2px 6px',
                    borderRadius: '4px',
                    background: s.bg,
                    color: s.color,
                    border: `1px solid ${s.border}`
                  }}>
                    {s.label}
                  </span>
                </div>

                <b style={{ display: 'block', fontSize: '0.85rem', fontWeight: 700, color: '#0f172a', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                  {n.name}
                </b>
              </div>

              <div style={{ marginTop: '10px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', marginBottom: '3px' }}>
                  <span style={{ color: '#64748b' }}>Stock / Cap</span>
                  <strong style={{ color: '#0f172a' }}>{Math.round(n.inventory)} <span style={{ color: '#94a3b8', fontWeight: 400 }}>/ {n.capacity}</span></strong>
                </div>

                {/* Progress bar */}
                <div style={{ width: '100%', height: '5px', background: '#e2e8f0', borderRadius: '3px', overflow: 'hidden', marginBottom: '8px' }}>
                  <div style={{ width: `${pct}%`, height: '100%', background: n.status === 'DISRUPTED' ? '#ef4444' : n.status === 'BUFFERING' ? '#0284c7' : '#10b981', borderRadius: '3px' }} />
                </div>

                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem' }}>
                  <span style={{ color: '#64748b' }}>Service</span>
                  <strong style={{ color: n.service_level < 0.9 ? '#d97706' : '#15803d' }}>
                    {(n.service_level * 100).toFixed(0)}%
                  </strong>
                </div>

                {n.status_detail && (
                  <p style={{ fontSize: '0.7rem', color: '#64748b', margin: '6px 0 0 0', lineHeight: '1.25' }}>
                    {n.status_detail}
                  </p>
                )}
              </div>
            </article>
          );
        })}
      </div>
    </section>
  );
}
