import {useState} from 'react';
import {create, step, negotiate, experiment, runFullSimulation, downloadRunReport} from './api';
import type {Disruption, State, ExperimentConfig, NetworkImport} from './types';
import Controls from './components/Controls';
import Network from './components/Network';
import NegotiationLog from './components/NegotiationLog';
import VerifierPanel from './components/VerifierPanel';
import Metrics from './components/Metrics';
import Comparison from './components/Comparison';
import RecoveryForecast from './components/RecoveryForecast';
import AgentRoster from './components/AgentRoster';
import AgentConversation from './components/AgentConversation';
import ARDNLab from './components/ARDNLab';
import './style.css';
import './forecast.css';
import './agents.css';
import './conversation.css';
import './navigation.css';
import './ardn-lab.css';
import './ardn-advanced.css';

export default function App(){
  const [d,setD]=useState<Disruption>({description:'',affected_nodes:[],affected_routes:[],effects:[],severity:.4,start_day:1,duration:5});
  const [experimentConfig,setExperimentConfig]=useState<ExperimentConfig>({severity:null,disruption_duration:null,simulation_horizon:12,max_negotiation_rounds:4,recovery_aggressiveness:'balanced'});
  const [network,setNetwork]=useState<NetworkImport|null>(null);
  const [state,setState]=useState<State|null>(null);
  const [results,setResults]=useState<any>();
  const [busy,setBusy]=useState(false);
  const [error,setError]=useState('');
  const [reportBusy,setReportBusy]=useState(false);
  const [page,setPage]=useState<'workspace'|'agents'|'ardn'>('workspace');

  async function run(){
    setBusy(true);setError('');
    try{
      const res = await runFullSimulation(d, experimentConfig, network);
      setState(res.state);
      setResults(res.experiment?.results || res.experiment);
    }catch(e:any){
      console.error(e);
      setError('We could not reach the local API. Start the backend on port 8000, then try again.');
    }finally{
      setBusy(false);
    }
  }

  async function downloadReport(){
    if(!state?.id)return;
    setReportBusy(true);setError('');
    try{await downloadRunReport(state.id);}catch(e){console.error(e);setError('The PDF report could not be generated.');}finally{setReportBusy(false);}
  }

  const isReady=Boolean(state);
  return <main className="app-shell">
    <a className="skip-link" href={page==='agents'?'#agent-room-title':page==='ardn'?'#ardn-lab-title':'#workspace'}>Skip to {page==='agents'?'agent room':page==='ardn'?'ARDN Lab':'workspace'}</a>
    <header className="topbar">
      <button className="brand brand-button" onClick={()=>setPage('workspace')} aria-label="Open Signal Chain workspace"><span className="brand-mark" aria-hidden="true">S</span><span>Signal Chain</span></button>
      <nav className="topnav" aria-label="Primary navigation"><button className={page==='workspace'?'active':''} onClick={()=>setPage('workspace')}>Workspace</button><button className={page==='agents'?'active':''} onClick={()=>setPage('agents')}>Agent room</button><button className={page==='ardn'?'active':''} onClick={()=>setPage('ardn')}>ARDN Lab</button></nav>
      <div className="topbar-meta"><span className="connection"><i aria-hidden="true"/> Local simulation</span><span className="agent-badge">{state?.mock_agents===false?(state.agent_model||'Qwen3.5 agents'):'Mock agents'}</span></div>
    </header>
    {error&&<p className="error-banner" role="alert"><strong>Unable to run scenario.</strong> {error}</p>}
    {page==='agents'?<AgentConversation state={state} onOpenWorkspace={()=>setPage('workspace')} onRun={run} busy={busy}/>:page==='ardn'?<ARDNLab onOpenWorkspace={()=>setPage('workspace')}/>:<>
    <section className="hero" aria-labelledby="page-title">
      <div><p className="eyebrow">SUPPLY CHAIN RESILIENCE LAB</p><h1 id="page-title">Plan a safer recovery.</h1><p className="lede">Test disruption scenarios, negotiate a response, and see what the safety verifier allows before a plan reaches your network.</p></div>
      <div className="hero-note"><span className="hero-note-icon" aria-hidden="true">✓</span><p><strong>Protected decisions</strong><br/>Every negotiated plan is checked against operating constraints.</p></div>
    </section>
    <Controls d={d} setD={setD} experiment={experimentConfig} setExperiment={setExperimentConfig} network={network} setNetwork={setNetwork} onRun={run} busy={busy}/>
    <section id="workspace" className={'workspace '+(!isReady?'workspace-empty':'')} aria-label="Simulation workspace">
      <div className="section-heading"><div><p className="eyebrow">LIVE WORKSPACE</p><h2>{isReady?'Recovery plan overview':'Your results will appear here'}</h2></div>{isReady&&<div style={{display:'flex',alignItems:'center',gap:'12px'}}><button onClick={downloadReport} disabled={reportBusy} style={{border:'1px solid #86efac',background:'#f0fdf4',color:'#166534',borderRadius:'6px',padding:'7px 11px',fontWeight:700,cursor:reportBusy?'wait':'pointer'}}>{reportBusy?'Preparing PDF...':'Download PDF report'}</button><span className="run-state"><i aria-hidden="true"/> Scenario complete {state?.meta&&` · Negotiation ${state.meta.rounds}/${state.meta.max_rounds||experimentConfig.max_negotiation_rounds}`}</span></div>}</div>
      {!isReady?<div className="empty-state"><div className="empty-illustration" aria-hidden="true"><span>●</span><b>→</b><span>●</span><b>→</b><span>●</span></div><h3>Start with a disruption scenario</h3><p>Adjust the inputs above and run the simulation to inspect network health, the proposed agreement, and its safety review.</p></div>:<div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
        {/* Row 1: Network Physical Flow & Scenario Pulse */}
        <div style={{ display: 'grid', gridTemplateColumns: '1.45fr 1fr', gap: '16px', alignItems: 'stretch' }}>
          <Network nodes={state?.nodes||[]} isDisruptionActive={state?.is_disruption_active} networkStatus={state?.network_status}/>
          <Metrics state={state}/>
        </div>

        {/* Row 2: Safety Review & Negotiation Trail */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1.35fr', gap: '16px', alignItems: 'stretch' }}>
          <VerifierPanel result={state?.verification}/>
          <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
            <AgentRoster active={isReady} real={state?.mock_agents===false}/>
            <NegotiationLog log={state?.negotiation||[]}/>
          </div>
        </div>

        {/* Row 3: ARDN Consequence Projection */}
        <RecoveryForecast forecast={state?.forecast}/>

        {/* Row 4: Multi-Strategy Benchmark Comparison */}
        <Comparison results={results} horizon={experimentConfig.simulation_horizon}/>
      </div>}
    </section>
    </>}
    <footer><span>Signal Chain · Decision sandbox</span><span>Results are simulated for research and planning.</span></footer>
  </main>
}
