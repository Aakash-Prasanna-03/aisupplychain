import {useState, useRef} from 'react';
import type {NetworkImport, ImportedNode} from '../types';

const TIER_OPTIONS = ['supplier','manufacturer','distributor','retailer'];

const TIER_COLORS: Record<string, string> = {
  supplier: '#3b9c72',
  manufacturer: '#3a7fc1',
  distributor: '#8a63d2',
  retailer: '#c17c3a',
};

const TIER_ALIASES: Record<string, string> = {
  vendor:'supplier', factory:'manufacturer', plant:'manufacturer',
  warehouse:'distributor', distribution_center:'distributor', store:'retailer', customer:'retailer',
};

function canonicalTier(t: string): string {
  return TIER_ALIASES[t.toLowerCase()] ?? t.toLowerCase();
}

const DEFAULT_NODE = (): ImportedNode => ({
  name:'', node_type:'supplier', inventory:100, capacity:200,
  production_capacity:50, holding_cost:0.2, shortage_cost:4, service_level_target:0.9,
});

const TEMPLATE_CSV = `name,node_type,inventory,capacity,production_capacity,holding_cost,shortage_cost,service_level_target
Raw Materials Supplier,supplier,500,800,300,0.15,5.00,0.95
Assembly Plant,manufacturer,300,600,250,0.35,6.00,0.95
Regional DC,distributor,200,400,0,0.25,5.00,0.95
Retail Network,retailer,150,300,0,0.40,12.00,0.98`;

interface Props {
  network: NetworkImport | null;
  onChange: (n: NetworkImport | null) => void;
}

export default function NetworkImportPanel({network, onChange}: Props) {
  const [open, setOpen] = useState(false);
  const [error, setError] = useState('');
  const [editIdx, setEditIdx] = useState<number | null>(null);
  const [editNode, setEditNode] = useState<ImportedNode>(DEFAULT_NODE());
  const [mode, setMode] = useState<'edit'|'json'|'csv'>('edit');
  const [rawJson, setRawJson] = useState('');
  const fileRef = useRef<HTMLInputElement>(null);

  const nodes = network?.nodes ?? [];

  // Full network validation — used only on bulk imports (file / paste)
  function validate(list: ImportedNode[]): string | null {
    if (list.length < 4) return 'Network must have at least 4 nodes.';
    const tiers = new Set(list.map(n => canonicalTier(n.node_type)));
    for (const t of ['supplier','manufacturer','distributor','retailer']) {
      if (!tiers.has(t)) return `Missing at least one node with tier: ${t}`;
    }
    for (const n of list) {
      if (!n.name.trim()) return 'All nodes need a name.';
      if (n.capacity <= 0) return `Node "${n.name}": capacity must be > 0`;
    }
    return null;
  }

  // Single-node validation — used when saving one node at a time
  function validateSingleNode(n: ImportedNode): string | null {
    if (!n.name.trim()) return 'Node needs a name.';
    if (n.capacity <= 0) return 'Capacity must be greater than 0.';
    return null;
  }

  // Whether the current node list meets the full network requirement (for status display)
  function networkStatus(list: ImportedNode[]): {ready: boolean; missingTiers: string[]} {
    const tiers = new Set(list.map(n => canonicalTier(n.node_type)));
    const missingTiers = ['supplier','manufacturer','distributor','retailer'].filter(t => !tiers.has(t));
    return { ready: list.length >= 4 && missingTiers.length === 0, missingTiers };
  }

  function applyNodes(list: ImportedNode[]) {
    const err = validate(list);
    if (err) { setError(err); return; }
    setError('');
    onChange({ nodes: list });
  }

  function parseCSV(text: string): ImportedNode[] {
    const lines = text.trim().split('\n').filter(Boolean);
    const headers = lines[0].split(',').map(h => h.trim());
    return lines.slice(1).map(line => {
      const vals = line.split(',').map(v => v.trim());
      const row: any = {};
      headers.forEach((h, i) => row[h] = vals[i] ?? '');
      return {
        name: row.name ?? '',
        node_type: (row.node_type ?? 'supplier') as ImportedNode['node_type'],
        inventory: parseFloat(row.inventory) || 0,
        capacity: parseFloat(row.capacity) || 0,
        production_capacity: parseFloat(row.production_capacity) || 0,
        holding_cost: parseFloat(row.holding_cost) || 0.2,
        shortage_cost: parseFloat(row.shortage_cost) || 4,
        service_level_target: parseFloat(row.service_level_target) || 0.9,
      };
    });
  }

  function onFileUpload(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0]; if (!file) return;
    const reader = new FileReader();
    reader.onload = ev => {
      const text = ev.target?.result as string;
      try {
        let list: ImportedNode[];
        if (file.name.endsWith('.json')) {
          const parsed = JSON.parse(text);
          list = Array.isArray(parsed) ? parsed : parsed.nodes ?? [];
        } else {
          list = parseCSV(text);
        }
        applyNodes(list);
      } catch { setError('Could not parse file. Check JSON/CSV format.'); }
    };
    reader.readAsText(file);
    e.target.value = '';
  }

  function onImportJson() {
    try {
      const parsed = JSON.parse(rawJson);
      const list: ImportedNode[] = Array.isArray(parsed) ? parsed : parsed.nodes ?? [];
      applyNodes(list);
      if (!validate(list)) { setMode('edit'); setRawJson(''); }
    } catch { setError('Invalid JSON. Check syntax and try again.'); }
  }

  function startEdit(i: number) { setEditIdx(i); setEditNode({...nodes[i]}); }
  function startAdd() { setEditIdx(nodes.length); setEditNode(DEFAULT_NODE()); }

  function saveEdit() {
    if (editIdx === null) return;
    // Only validate the individual node — NOT the full network
    const nodeErr = validateSingleNode(editNode);
    if (nodeErr) { setError(nodeErr); return; }
    setError('');
    const list = [...nodes];
    list[editIdx] = editNode;
    // Store directly — network completeness is checked separately as a status
    onChange({ nodes: list });
    setEditIdx(null);
  }

  function removeNode(i: number) {
    const list = nodes.filter((_, idx) => idx !== i);
    onChange(list.length ? { nodes: list } : null);
  }

  function downloadTemplate(fmt: 'json'|'csv') {
    const blob = fmt === 'csv'
      ? new Blob([TEMPLATE_CSV], {type:'text/csv'})
      : new Blob([JSON.stringify({nodes:[
          {name:'Raw Materials Supplier',node_type:'supplier',inventory:500,capacity:800,production_capacity:300,holding_cost:0.15,shortage_cost:5,service_level_target:0.95},
          {name:'Assembly Plant',node_type:'manufacturer',inventory:300,capacity:600,production_capacity:250,holding_cost:0.35,shortage_cost:6,service_level_target:0.95},
          {name:'Regional DC',node_type:'distributor',inventory:200,capacity:400,production_capacity:0,holding_cost:0.25,shortage_cost:5,service_level_target:0.95},
          {name:'Retail Network',node_type:'retailer',inventory:150,capacity:300,production_capacity:0,holding_cost:0.40,shortage_cost:12,service_level_target:0.98},
        ]},null,2)], {type:'application/json'});
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `network-template.${fmt}`;
    a.click();
  }

  const tierCounts = TIER_OPTIONS.reduce((acc, t) => {
    acc[t] = nodes.filter(n => canonicalTier(n.node_type) === t).length;
    return acc;
  }, {} as Record<string,number>);

  const { ready: networkReady, missingTiers } = networkStatus(nodes);

  return (
    <div className="ni-root">
      <button
        className={`ni-toggle ${network ? 'ni-toggle--active' : ''}`}
        onClick={() => setOpen(o => !o)}
        aria-expanded={open}
      >
        <span className="ni-toggle-icon">{network ? '✓' : '⊕'}</span>
        <span className="ni-toggle-label">
          {network
            ? `Custom network · ${nodes.length} node${nodes.length === 1 ? '' : 's'}`
            : 'Import custom network'}
        </span>
        {network && (
          <button className="ni-clear-btn" onClick={e => { e.stopPropagation(); onChange(null); setOpen(false); }} title="Remove custom network">
            ✕
          </button>
        )}
        <span className="ni-chevron">{open ? '▲' : '▼'}</span>
      </button>

      {open && (
        <div className="ni-panel" role="region" aria-label="Custom network import">
          <div className="ni-header">
            <div>
              <p className="ni-eyebrow">CUSTOM SUPPLY CHAIN</p>
              <h3 className="ni-title">Define your network</h3>
              <p className="ni-subtitle">Override the default benchmark with your own nodes and cost parameters.</p>
            </div>
            <div className="ni-header-actions">
              <button className="ni-dl-btn" onClick={() => downloadTemplate('csv')} title="Download CSV template">↓ CSV</button>
              <button className="ni-dl-btn" onClick={() => downloadTemplate('json')} title="Download JSON template">↓ JSON</button>
              <button className="ni-upload-btn" onClick={() => fileRef.current?.click()}>
                <span>⤒</span> Upload file
              </button>
              <input ref={fileRef} type="file" accept=".json,.csv" style={{display:'none'}} onChange={onFileUpload}/>
            </div>
          </div>

          {/* Tier coverage indicators */}
          <div className="ni-tiers">
            {TIER_OPTIONS.map(t => (
              <div key={t} className={`ni-tier ${tierCounts[t] > 0 ? 'ni-tier--ok' : 'ni-tier--missing'}`}>
                <span className="ni-tier-dot" style={{background: TIER_COLORS[t]}}/>
                <span className="ni-tier-name">{t}</span>
                <span className="ni-tier-count">{tierCounts[t] || '⚠ missing'}</span>
              </div>
            ))}
          </div>

          {error && <p className="ni-error" role="alert">⚠ {error}</p>}

          {/* Tab bar */}
          <div className="ni-tabs">
            {(['edit','json','csv'] as const).map(m => (
              <button key={m} className={`ni-tab ${mode === m ? 'ni-tab--active' : ''}`} onClick={() => { setMode(m); setError(''); }}>
                {m === 'edit' ? 'Node editor' : m === 'json' ? 'Paste JSON' : 'Paste CSV'}
              </button>
            ))}
          </div>

          {mode === 'edit' && (
            <div className="ni-editor">
              {nodes.length > 0 ? (
                <div className="ni-node-list">
                  {nodes.map((n, i) => (
                    <div key={i} className="ni-node-row">
                      <span className="ni-node-dot" style={{background: TIER_COLORS[canonicalTier(n.node_type)] ?? '#999'}}/>
                      <div className="ni-node-info">
                        <strong>{n.name || <em>Unnamed</em>}</strong>
                        <small>{n.node_type} · inv {n.inventory} · cap {n.capacity} · hold ${n.holding_cost}/u · short ${n.shortage_cost}/u</small>
                      </div>
                      <div className="ni-node-actions">
                        <button className="ni-action-btn" onClick={() => startEdit(i)}>Edit</button>
                        <button className="ni-action-btn ni-action-btn--del" onClick={() => removeNode(i)}>✕</button>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="ni-empty">
                  <p>No nodes yet. Upload a file or add nodes manually.</p>
                </div>
              )}

              <button className="ni-add-btn" onClick={startAdd}>+ Add node</button>

              {editIdx !== null && (
                <div className="ni-edit-form">
                  <div className="ni-edit-form-header">
                    <strong>{editIdx < nodes.length ? 'Edit node' : 'Add node'}</strong>
                    <button className="ni-close-form" onClick={() => setEditIdx(null)}>✕</button>
                  </div>
                  <div className="ni-form-grid">
                    <div className="ni-form-field ni-form-field--wide">
                      <label>Node name</label>
                      <input type="text" value={editNode.name} onChange={e => setEditNode({...editNode, name:e.target.value})} placeholder="e.g. Main Assembly Plant"/>
                    </div>
                    <div className="ni-form-field">
                      <label>Tier / Node type</label>
                      <select value={editNode.node_type} onChange={e => setEditNode({...editNode, node_type:e.target.value as ImportedNode['node_type']})}>
                        {['supplier','vendor','manufacturer','factory','plant','distributor','warehouse','distribution_center','retailer','store','customer'].map(t => (
                          <option key={t} value={t}>{t}</option>
                        ))}
                      </select>
                    </div>
                    <div className="ni-form-field">
                      <label>Inventory (units)</label>
                      <input type="number" min="0" value={editNode.inventory} onChange={e => setEditNode({...editNode, inventory:+e.target.value})}/>
                    </div>
                    <div className="ni-form-field">
                      <label>Capacity (max units)</label>
                      <input type="number" min="1" value={editNode.capacity} onChange={e => setEditNode({...editNode, capacity:+e.target.value})}/>
                    </div>
                    <div className="ni-form-field">
                      <label>Production capacity</label>
                      <input type="number" min="0" value={editNode.production_capacity} onChange={e => setEditNode({...editNode, production_capacity:+e.target.value})}/>
                    </div>
                    <div className="ni-form-field">
                      <label>Holding cost ($/unit/day)</label>
                      <input type="number" min="0" step="0.01" value={editNode.holding_cost} onChange={e => setEditNode({...editNode, holding_cost:+e.target.value})}/>
                    </div>
                    <div className="ni-form-field">
                      <label>Shortage cost ($/unit)</label>
                      <input type="number" min="0" step="0.01" value={editNode.shortage_cost} onChange={e => setEditNode({...editNode, shortage_cost:+e.target.value})}/>
                    </div>
                    <div className="ni-form-field">
                      <label>Service level target</label>
                      <input type="number" min="0.1" max="1" step="0.01" value={editNode.service_level_target} onChange={e => setEditNode({...editNode, service_level_target:+e.target.value})}/>
                    </div>
                  </div>
                  <div className="ni-form-footer">
                    <button className="ni-cancel-btn" onClick={() => setEditIdx(null)}>Cancel</button>
                    <button className="ni-save-btn" onClick={saveEdit}>Save node</button>
                  </div>
                </div>
              )}

              {/* Network readiness status banner */}
              {nodes.length > 0 && (
                networkReady ? (
                  <div className="ni-valid-banner">
                    <span>✓</span> Network is valid — all 4 tiers covered. Will be used in the simulation.
                  </div>
                ) : (
                  <div className="ni-progress-banner">
                    <span>◐</span> Keep adding nodes — still need: {missingTiers.join(', ')}{nodes.length < 4 ? ` (${4 - nodes.length} more node${4 - nodes.length === 1 ? '' : 's'} needed)` : ''}
                  </div>
                )
              )}
            </div>
          )}

          {mode === 'json' && (
            <div className="ni-paste-area">
              <p className="ni-paste-hint">Paste a JSON object with a <code>nodes</code> array, or a raw array of node objects.</p>
              <textarea className="ni-textarea" rows={10} value={rawJson} onChange={e => setRawJson(e.target.value)} placeholder={'{\n  "nodes": [\n    {"name":"Supplier 1","node_type":"supplier",...}\n  ]\n}'}/>
              <div className="ni-paste-footer">
                <button className="ni-cancel-btn" onClick={() => setMode('edit')}>Cancel</button>
                <button className="ni-save-btn" onClick={onImportJson}>Import JSON</button>
              </div>
            </div>
          )}

          {mode === 'csv' && (
            <div className="ni-paste-area">
              <p className="ni-paste-hint">Paste CSV with header: <code>name,node_type,inventory,capacity,production_capacity,holding_cost,shortage_cost,service_level_target</code></p>
              <textarea className="ni-textarea" rows={10} value={rawJson} onChange={e => setRawJson(e.target.value)} placeholder={TEMPLATE_CSV}/>
              <div className="ni-paste-footer">
                <button className="ni-cancel-btn" onClick={() => setMode('edit')}>Cancel</button>
                <button className="ni-save-btn" onClick={() => {
                  try {
                    const list = parseCSV(rawJson);
                    applyNodes(list);
                    if (!validate(list)) { setMode('edit'); setRawJson(''); }
                  } catch { setError('Could not parse CSV. Check format.'); }
                }}>Import CSV</button>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
