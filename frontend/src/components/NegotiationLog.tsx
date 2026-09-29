import React, { useState } from 'react';

export default function NegotiationLog({ log }: { log: any[] }) {
  const [activeTab, setActiveTab] = useState<'approved' | 'all'>('all');

  const verifierEvents = log.filter(x => x.speaker === 'Verifier');
  const finalApproval = verifierEvents.find(x => x.valid);

  const getActorBadge = (speaker: string) => {
    if (speaker.includes('Supplier')) return { bg: '#ecfdf5', color: '#047857', border: '#a7f3d0' };
    if (speaker.includes('Manufacturer')) return { bg: '#f5f3ff', color: '#6d28d9', border: '#ddd6fe' };
    if (speaker.includes('Distributor')) return { bg: '#fffbeb', color: '#b45309', border: '#fde68a' };
    if (speaker.includes('Retailer')) return { bg: '#fff1f2', color: '#be123c', border: '#fecdd3' };
    return { bg: '#f1f5f9', color: '#334155', border: '#cbd5e1' };
  };

  return (
    <section className="panel log-panel" aria-labelledby="negotiation-title" style={{ height: '100%' }}>
      <div className="panel-heading" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '14px' }}>
        <div>
          <h2 id="negotiation-title" style={{ fontSize: '1rem', fontWeight: 700, color: '#173b36', margin: 0 }}>
            Multi-Agent Negotiation Trail
          </h2>
          <p style={{ fontSize: '0.8rem', color: '#64748b', margin: '2px 0 0 0' }}>
            Turn-by-turn counterproposals and verifier-guided revision rounds
          </p>
        </div>

        <div style={{ display: 'flex', gap: '4px', background: '#f1f5f9', padding: '3px', borderRadius: '6px' }}>
          <button
            onClick={() => setActiveTab('all')}
            style={{
              padding: '4px 10px',
              borderRadius: '4px',
              border: 0,
              fontSize: '0.75rem',
              fontWeight: 600,
              background: activeTab === 'all' ? '#fff' : 'transparent',
              color: activeTab === 'all' ? '#0f172a' : '#64748b',
              boxShadow: activeTab === 'all' ? '0 1px 3px rgba(0,0,0,0.1)' : 'none'
            }}
          >
            Full Trail ({log.length})
          </button>
          <button
            onClick={() => setActiveTab('approved')}
            style={{
              padding: '4px 10px',
              borderRadius: '4px',
              border: 0,
              fontSize: '0.75rem',
              fontWeight: 600,
              background: activeTab === 'approved' ? '#fff' : 'transparent',
              color: activeTab === 'approved' ? '#15803d' : '#64748b',
              boxShadow: activeTab === 'approved' ? '0 1px 3px rgba(0,0,0,0.1)' : 'none'
            }}
          >
            Final Agreement
          </button>
        </div>
      </div>

      {activeTab === 'approved' ? (
        <div style={{ padding: '16px', background: '#f0fdf4', border: '1px solid #bbf7d0', borderRadius: '8px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px' }}>
            <span style={{ width: '8px', height: '8px', borderRadius: '50%', background: '#16a34a' }} />
            <strong style={{ color: '#14532d', fontSize: '0.9rem' }}>Executed Multi-Tier Agreement</strong>
          </div>
          <p style={{ fontSize: '0.85rem', color: '#166534', margin: 0, lineHeight: '1.5' }}>
            {finalApproval?.message || 'Final compliant agreement reached and approved for execution across all tiers.'}
          </p>
        </div>
      ) : (
        <div style={{
          maxHeight: '360px',
          overflowY: 'auto',
          display: 'flex',
          flexDirection: 'column',
          gap: '8px',
          paddingRight: '4px'
        }}>
          {log.map((x, i) => {
            const isVerifier = x.speaker === 'Verifier';
            const badge = getActorBadge(x.speaker);

            if (isVerifier) {
              const isApproved = x.valid;
              return (
                <div
                  key={i}
                  style={{
                    padding: '10px 14px',
                    borderRadius: '8px',
                    background: isApproved ? '#f0fdf4' : '#fef2f2',
                    border: `1px solid ${isApproved ? '#bbf7d0' : '#fecaca'}`,
                    display: 'flex',
                    alignItems: 'flex-start',
                    gap: '10px'
                  }}
                >
                  <span style={{
                    width: '20px',
                    height: '20px',
                    borderRadius: '50%',
                    background: isApproved ? '#16a34a' : '#dc2626',
                    color: '#fff',
                    display: 'grid',
                    placeItems: 'center',
                    fontSize: '0.75rem',
                    fontWeight: 800,
                    flexShrink: 0,
                    marginTop: '1px'
                  }}>
                    {isApproved ? '✓' : '!'}
                  </span>
                  <div style={{ flex: 1 }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '2px' }}>
                      <b style={{ fontSize: '0.8rem', color: isApproved ? '#14532d' : '#7f1d1d' }}>
                        Verifier Gate · {isApproved ? 'Approved' : 'Revision Enforced'}
                      </b>
                      <span style={{ fontSize: '0.7rem', color: '#64748b' }}>Check #{i + 1}</span>
                    </div>
                    <p style={{ margin: 0, fontSize: '0.8rem', color: isApproved ? '#166534' : '#991b1b', lineHeight: '1.4' }}>
                      {x.message}
                    </p>
                  </div>
                </div>
              );
            }

            return (
              <div
                key={i}
                style={{
                  padding: '10px 12px',
                  borderRadius: '8px',
                  background: '#f8fafc',
                  border: '1px solid #e2e8f0',
                  display: 'flex',
                  gap: '10px'
                }}
              >
                <div style={{ flex: 1 }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '4px' }}>
                    <span style={{
                      fontSize: '0.7rem',
                      fontWeight: 700,
                      padding: '2px 8px',
                      borderRadius: '4px',
                      background: badge.bg,
                      color: badge.color,
                      border: `1px solid ${badge.border}`
                    }}>
                      {x.speaker}
                    </span>
                    <span style={{ fontSize: '0.7rem', color: '#94a3b8' }}>Event #{i + 1}</span>
                  </div>
                  <p style={{ margin: 0, fontSize: '0.8rem', color: '#334155', lineHeight: '1.4' }}>
                    {x.message}
                  </p>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </section>
  );
}
