import {useState} from 'react';
import {interpretScenario} from '../api';
import type {Disruption,ExperimentConfig} from '../types';

export default function Controls({d,setD,experiment,setExperiment,onRun,busy}:{d:Disruption;setD:(d:Disruption)=>void;experiment:ExperimentConfig;setExperiment:(c:ExperimentConfig)=>void;onRun:()=>void;busy:boolean}){
  const [prompt,setPrompt]=useState('');
  const [interpretBusy,setInterpretBusy]=useState(false);
  const [message,setMessage]=useState('');
  const [source,setSource]=useState('');
  const [fallbackReason,setFallbackReason]=useState('');
  const [error,setError]=useState('');
  async function generate(){
    if(!prompt.trim()) return;
    setInterpretBusy(true);setError('');
    try{const result=await interpretScenario(prompt,experiment);setD(result.scenario);setExperiment(result.experiment);setMessage(result.message);setSource(result.source);setFallbackReason(result.fallback_reason||'');}
    catch{setError('I could not interpret that scenario. Check that the backend is running, then try again.');}
    finally{setInterpretBusy(false);}
  }
  const update=(change:Partial<ExperimentConfig>)=>setExperiment({...experiment,...change});
  return <section className="scenario-card" aria-labelledby="scenario-title">
    <div className="scenario-heading"><div><p className="eyebrow">DISRUPTION SCENARIO</p><h2 id="scenario-title">Describe what happened</h2></div><p>Use natural language to create any supply-chain scenario.</p></div>
    <div className="scenario-prompt">
      <label htmlFor="scenario-prompt-input">What would you like to simulate?</label>
      <div className="prompt-row"><textarea id="scenario-prompt-input" rows={3} value={prompt} onChange={e=>setPrompt(e.target.value)} onKeyDown={e=>{if((e.ctrlKey||e.metaKey)&&e.key==='Enter')generate();}} placeholder="Example: A fire at the supplier's factory reduces production capacity by 60% for four days."/><button className="generate-button" onClick={generate} disabled={interpretBusy||!prompt.trim()}>{interpretBusy?'Interpreting…':'Interpret scenario'}</button></div>
      <div className="prompt-footer"><span>Describe the event, affected operations, timing, and impact in your own words.</span>{message&&<span className="interpretation">✓ {message}</span>}</div>
      {error&&<p className="prompt-error" role="alert">{error}</p>}
    </div>
    <details className="advanced-controls"><summary>Advanced controls <span>Configure how the experiment runs</span></summary>
      <div className="generic-controls">
        <div className="field severity-field"><div className="label-row"><label htmlFor="severity-control">Severity</label><output htmlFor="severity-control">{experiment.severity===null?`${Math.round(d.severity*100)}% inferred`: `${experiment.severity}%`}</output></div><input id="severity-control" type="range" min="10" max="100" step="1" value={experiment.severity??Math.round(d.severity*100)} onChange={e=>update({severity:+e.target.value})}/><div className="range-labels"><span>10% impact</span><span>100% impact</span></div></div>
        <div className="field"><label htmlFor="duration-control">Disruption duration</label><select id="duration-control" value={experiment.disruption_duration??'automatic'} onChange={e=>update({disruption_duration:e.target.value==='automatic'?null:+e.target.value})}><option value="automatic">Automatic from scenario</option>{[1,2,3,4,5,6,7].map(day=><option value={day} key={day}>{day} days</option>)}</select></div>
        <div className="field"><label htmlFor="horizon-control">Simulation horizon</label><div className="input-suffix"><input id="horizon-control" type="number" min="4" max="14" value={experiment.simulation_horizon} onChange={e=>update({simulation_horizon:+e.target.value})}/><span>days</span></div></div>
        <div className="field"><label htmlFor="rounds-control">Maximum negotiation rounds</label><input id="rounds-control" type="number" min="1" max="4" value={experiment.max_negotiation_rounds} onChange={e=>update({max_negotiation_rounds:+e.target.value})}/></div>
        <div className="field"><label htmlFor="aggressiveness-control">Recovery aggressiveness</label><select id="aggressiveness-control" value={experiment.recovery_aggressiveness} onChange={e=>update({recovery_aggressiveness:e.target.value as ExperimentConfig['recovery_aggressiveness']})}><option value="conservative">Conservative</option><option value="balanced">Balanced</option><option value="aggressive">Aggressive</option></select></div>
      </div>
    </details>
    {d.description&&<div className="scenario-interpretation" aria-live="polite"><p className="interpretation-title">SCENARIO INTERPRETATION</p><strong>{d.description}</strong><div className="interpretation-grid"><span><small>Affected nodes</small>{d.affected_nodes.join(', ')||'Inferred from scenario'}</span><span><small>Affected routes</small>{d.affected_routes.length?d.affected_routes.map(route=>`${route.from_node||route.source} → ${route.to_node||route.destination}`).join('; '):'None identified'}</span><span><small>Operational effects</small>{d.effects.map(effect=>(effect.description||effect.type)+(effect.magnitude>0?` (${Math.round(effect.magnitude*100)}%)`:'' )).join('; ')}</span><span><small>Estimated impact</small>{experiment.severity??Math.round(d.severity*100)}% overall severity · Day {d.start_day} for {d.duration} days</span></div><div className="interpretation-source"><span>Interpretation source: <strong>{source==='llm'?'Gemini':'Local fallback parser'}</strong></span>{fallbackReason&&<span>Reason: {fallbackReason}</span>}</div></div>}
    <button className="run-button scenario-run" onClick={onRun} disabled={busy||!d.description} aria-busy={busy}><span aria-hidden="true">{busy?'◌':'▶'}</span>{busy?'Running scenario…':'Run scenario'}</button>
  </section>
}
