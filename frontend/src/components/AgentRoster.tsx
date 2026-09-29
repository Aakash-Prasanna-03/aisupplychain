import React from 'react';

const coordinators = [
  { name: 'Supplier', tier: 'Upstream Supply', accent: '#047857', bg: '#ecfdf5', border: '#a7f3d0' },
  { name: 'Manufacturer', tier: 'Production Facility', accent: '#6d28d9', bg: '#f5f3ff', border: '#ddd6fe' },
  { name: 'Distributor', tier: 'Logistics Warehouse', accent: '#b45309', bg: '#fffbeb', border: '#fde68a' },
  { name: 'Retailer', tier: 'Store Fulfillment', accent: '#be123c', bg: '#fff1f2', border: '#fecdd3' },
];

export default function AgentRoster({ active, real }: { active: boolean; real: boolean }) {
  return (
    <div style={{
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-between',
      background: '#fff',
      border: '1px solid #e2e8f0',
      borderRadius: '8px',
      padding: '8px 14px',
      marginBottom: '12px',
      flexWrap: 'wrap',
      gap: '8px'
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
        <span style={{ fontSize: '0.7rem', fontWeight: 800, textTransform: 'uppercase', letterSpacing: '0.05em', color: '#173b36' }}>
          Negotiation Team ({real ? 'Autonomous LLM' : 'Fallback Agents'}):
        </span>
      </div>

      <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
        {coordinators.map((c) => (
          <div
            key={c.name}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              padding: '3px 8px',
              borderRadius: '6px',
              background: c.bg,
              border: `1px solid ${c.border}`,
              fontSize: '0.75rem',
              fontWeight: 600,
              color: c.accent
            }}
          >
            <span style={{ width: '6px', height: '6px', borderRadius: '50%', background: c.accent }} />
            <span>{c.name} Coordinator</span>
            <span style={{ fontSize: '0.65rem', color: '#64748b', fontWeight: 400 }}>({c.tier})</span>
          </div>
        ))}
      </div>
    </div>
  );
}
