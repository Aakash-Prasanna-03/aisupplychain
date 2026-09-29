import {useState} from 'react';
import {interpretScenario} from '../api';
import type {Disruption,ExperimentConfig,NetworkImport} from '../types';
import NetworkImportPanel from './NetworkImportPanel';

interface Props {
  d: Disruption;
  setD: (d: Disruption) => void;
  experiment: ExperimentConfig;
  setExperiment: (c: ExperimentConfig) => void;
  network: NetworkImport | null;
  setNetwork: (n: NetworkImport | null) => void;
  onRun: () => void;
  busy: boolean;
}

export default function Controls({d,setD,experiment,setExperiment,network,setNetwork,onRun,busy}: Props){
  const [prompt,setPrompt]=useState('');
  const [interpretBusy,setInterpretBusy]=useState(false);
  const [message,setMessage]=useState('');
  const [source,setSource]=useState('');
  const [fallbackReason,setFallbackReason]=useState('');
  const [error,setError]=useState('');
  const [showAdvanced,setShowAdvanced]=useState(false);

  async function generate(){
    if(!prompt.trim()) return;
    setInterpretBusy(true);setError('');
    try{
      const result=await interpretScenario(prompt,experiment);
      setD(result.scenario);setExperiment(result.experiment);
      setMessage(result.message);setSource(result.source);
      setFallbackReason(result.fallback_reason||'');
    }
    catch{setError('Could not interpret that scenario. Check that the backend is running.');}
    finally{setInterpretBusy(false);}
  }

  const update=(change:Partial<ExperimentConfig>)=>setExperiment({...experiment,...change});

  const canRun = Boolean(d.description) && !busy;

  return (
    <section className="sc2-card" aria-labelledby="scenario-title">

      {/* ── Header ── */}
      <div className="sc2-header">
        <div className="sc2-header-left">
          <p className="sc2-eyebrow">DISRUPTION SCENARIO</p>
          <h2 id="scenario-title" className="sc2-title">Describe what happened</h2>
        </div>
        <p className="sc2-tagline">Use natural language to create any supply-chain scenario.</p>
      </div>

      {/* ── Prompt area ── */}
      <div className="sc2-prompt-wrap">
        <label className="sc2-prompt-label" htmlFor="sc2-prompt">What would you like to simulate?</label>
        <div className="sc2-prompt-row">
          <textarea
            id="sc2-prompt"
            className="sc2-textarea"
            rows={3}
            value={prompt}
            onChange={e=>setPrompt(e.target.value)}
            onKeyDown={e=>{if((e.ctrlKey||e.metaKey)&&e.key==='Enter')generate();}}
            placeholder="Example: A fire at the supplier's factory reduces production capacity by 60% for four days."
          />
          <button
            className="sc2-interpret-btn"
            onClick={generate}
            disabled={interpretBusy||!prompt.trim()}
          >
            {interpretBusy
              ? <><span className="sc2-spin">◌</span> Interpreting…</>
              : 'Interpret scenario'}
          </button>
        </div>
        <div className="sc2-prompt-footer">
          <span className="sc2-hint">Describe the event, affected operations, timing, and impact in your own words.</span>
          {message && <span className="sc2-confirmed">✓ {message}</span>}
        </div>
        {error && <p className="sc2-error" role="alert">{error}</p>}
      </div>

      {/* ── Scenario interpretation card ── */}
      {d.description && (
        <div className="sc2-interp" aria-live="polite">
          <div className="sc2-interp-header">
            <span className="sc2-interp-badge">SCENARIO INTERPRETATION</span>
            <span className={`sc2-source-pill ${source==='llm'?'sc2-source-pill--llm':''}`}>
              {source==='llm'?'✦ Gemini':'Local parser'}
            </span>
          </div>
          <p className="sc2-interp-desc">{d.description}</p>
          <div className="sc2-interp-grid">
            <div className="sc2-interp-block">
              <small>Affected nodes</small>
              <span>{d.affected_nodes.join(', ')||'Inferred from scenario'}</span>
            </div>
            <div className="sc2-interp-block">
              <small>Affected routes</small>
              <span>{d.affected_routes.length
                ? d.affected_routes.map(r=>`${r.from_node||r.source} → ${r.to_node||r.destination}`).join('; ')
                : 'None identified'}</span>
            </div>
            <div className="sc2-interp-block sc2-interp-block--wide">
              <small>Operational effects</small>
              <span>{d.effects.map(e=>(e.description||e.type)+(e.magnitude>0?` (${Math.round(e.magnitude*100)}%)`:'') ).join(' · ')||'—'}</span>
            </div>
            <div className="sc2-interp-block">
              <small>Estimated impact</small>
              <span>{experiment.severity??Math.round(d.severity*100)}% severity · Day {d.start_day} for {d.duration} days</span>
            </div>
          </div>
          {fallbackReason && <p className="sc2-fallback-note">Fallback reason: {fallbackReason}</p>}
        </div>
      )}

      {/* ── Advanced controls (collapsible) ── */}
      <div className="sc2-advanced">
        <button
          className="sc2-advanced-toggle"
          onClick={()=>setShowAdvanced(a=>!a)}
          aria-expanded={showAdvanced}
        >
          <span className="sc2-adv-icon">{showAdvanced?'−':'+'}</span>
          Advanced controls
          <span className="sc2-adv-sub">Configure how the experiment runs</span>
        </button>

        {showAdvanced && (
          <div className="sc2-advanced-body">
            <div className="sc2-field sc2-field--range">
              <div className="sc2-field-row">
                <label htmlFor="sc2-severity">Severity</label>
                <output htmlFor="sc2-severity" className="sc2-output">
                  {experiment.severity===null?`${Math.round(d.severity*100)}% inferred`:`${experiment.severity}%`}
                </output>
              </div>
              <input id="sc2-severity" type="range" min="10" max="100" step="1"
                value={experiment.severity??Math.round(d.severity*100)}
                onChange={e=>update({severity:+e.target.value})}/>
              <div className="sc2-range-labels"><span>10%</span><span>100%</span></div>
            </div>
            <div className="sc2-field">
              <label htmlFor="sc2-duration">Disruption duration</label>
              <select id="sc2-duration" value={experiment.disruption_duration??'automatic'}
                onChange={e=>update({disruption_duration:e.target.value==='automatic'?null:+e.target.value})}>
                <option value="automatic">Automatic from scenario</option>
                {[1,2,3,4,5,6,7].map(d=><option value={d} key={d}>{d} days</option>)}
              </select>
            </div>
            <div className="sc2-field">
              <label htmlFor="sc2-horizon">Simulation horizon</label>
              <div className="sc2-input-suffix">
                <input id="sc2-horizon" type="number" min="4" max="14"
                  value={experiment.simulation_horizon}
                  onChange={e=>update({simulation_horizon:+e.target.value})}/>
                <span>days</span>
              </div>
            </div>
            <div className="sc2-field">
              <label htmlFor="sc2-rounds">Max negotiation rounds</label>
              <input id="sc2-rounds" type="number" min="1" max="4"
                value={experiment.max_negotiation_rounds}
                onChange={e=>update({max_negotiation_rounds:+e.target.value})}/>
            </div>
            <div className="sc2-field">
              <label htmlFor="sc2-agg">Recovery aggressiveness</label>
              <select id="sc2-agg" value={experiment.recovery_aggressiveness}
                onChange={e=>update({recovery_aggressiveness:e.target.value as ExperimentConfig['recovery_aggressiveness']})}>
                <option value="conservative">Conservative</option>
                <option value="balanced">Balanced</option>
                <option value="aggressive">Aggressive</option>
              </select>
            </div>
          </div>
        )}
      </div>

      {/* ── Network import panel ── */}
      <NetworkImportPanel network={network} onChange={setNetwork}/>

      {/* ── Footer / Run ── */}
      <div className="sc2-footer">
        {network && (
          <span className="sc2-network-badge">
            <span className="sc2-nb-dot"/>
            Custom network · {network.nodes.length} nodes active
          </span>
        )}
        <button
          className="sc2-run-btn"
          onClick={onRun}
          disabled={!canRun}
          aria-busy={busy}
        >
          {busy
            ? <><span className="sc2-spin">◌</span> Running scenario…</>
            : <><span>▶</span> Run scenario</>}
        </button>
      </div>

    </section>
  );
}
