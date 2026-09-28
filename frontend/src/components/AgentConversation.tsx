import type { State } from '../types';

type ConversationEvent={speaker:string;message:string;valid?:boolean;violations:any[];proposal?:any};

const people=[
  {key:'Supplier',name:'Supplier agent',role:'Capacity and upstream continuity',initial:'S',accent:'mint'},
  {key:'Manufacturer',name:'Manufacturer agent',role:'Production flow and material balance',initial:'M',accent:'violet'},
  {key:'Distributor',name:'Distributor agent',role:'Network allocation and availability',initial:'D',accent:'amber'},
  {key:'Retailer',name:'Retailer agent',role:'Customer service and demand impact',initial:'R',accent:'coral'},
];

const preview:ConversationEvent[]=[
  {speaker:'Supplier Agent',message:'I can protect upstream recovery capacity while keeping the manufacturer supplied.',violations:[]},
  {speaker:'Manufacturer Agent',message:'I will balance inbound flow with production limits before committing downstream volume.',violations:[]},
  {speaker:'Distributor Agent',message:'I will preserve inventory where it reduces the largest downstream service risk.',violations:[]},
  {speaker:'Retailer Agent',message:'I will keep customer-service impact visible while the network chooses a safe plan.',violations:[]},
  {speaker:'Verifier',message:'Every proposed agreement is checked for inventory, capacity, service, and fair allocation before execution.',valid:true,violations:[]},
];

function actorFor(speaker:string){return people.find(person=>person.key===speaker.replace(' (Mock)','').replace(' Agent',''));}
function shipmentSummary(proposal:any){const shipments=proposal?.shipments||[];return shipments.length?shipments.map((shipment:any)=>`${shipment.from} to ${shipment.to}: ${shipment.quantity}`).join('  ·  '):null;}

export default function AgentConversation({state,onOpenWorkspace,onRun,busy}:{state:State|null;onOpenWorkspace:()=>void;onRun:()=>void;busy:boolean}){
  const hasRun=Boolean(state?.negotiation?.length);
  const events:ConversationEvent[]=hasRun?state!.negotiation:preview;
  const isMock=state?.mock_agents!==false;
  const verifierEvents=events.filter(event=>event.speaker==='Verifier');
  const revisions=verifierEvents.filter(event=>event.valid===false).length;
  const approved=Boolean(state?.verification?.valid);
  return <section className="agent-room" aria-labelledby="agent-room-title">
    <div className="agent-room-hero">
      <div className="agent-room-copy"><p className="eyebrow">MULTI-AGENT DECISION ROOM</p><h1 id="agent-room-title">Let every part of the supply chain have a voice.</h1><p>Signal Chain brings operational priorities into one traceable decision thread, then lets a deterministic safety layer decide what can move forward.</p><div className="agent-room-actions"><button className="agent-primary" onClick={onRun} disabled={busy}>{busy?'Running live scenario…':hasRun?'Run another scenario':'Run a live conversation'}</button><button className="agent-secondary" onClick={onOpenWorkspace}>Configure scenario</button></div></div>
      <aside className="trust-card" aria-label="Decision safeguards"><span className="trust-kicker">DECISION SAFEGUARDS</span><strong>Four perspectives.<br/>One accountable plan.</strong><ul><li><span aria-hidden="true">01</span> Role-bounded proposals</li><li><span aria-hidden="true">02</span> Verifier-guided revision</li><li><span aria-hidden="true">03</span> Auditable decision trail</li></ul></aside>
    </div>
    <div className="room-status" role="status"><span className={'status-dot '+(hasRun?'is-live':'is-preview')} aria-hidden="true"/><div><b>{hasRun?'Scenario transcript':'Conversation preview'}</b><p>{hasRun?(isMock?'This run used deterministic fallback agents.':'This run used the configured Qwen agents.'):'This is an illustrative preview of the workflow, not a model transcript.'}</p></div>{hasRun&&<span className={'approval-chip '+(approved?'is-approved':'')}>{approved?'Verifier approved':'Verifier review required'}</span>}</div>
    <div className="agent-people" aria-label="Negotiating agents">{people.map(person=><article key={person.key} className={'person-card '+person.accent}><span className="person-avatar" aria-hidden="true">{person.initial}</span><div><b>{person.name}</b><p>{person.role}</p></div><span className="person-presence" aria-label="Available"/></article>)}</div>
    <div className="conversation-layout">
      <section className="conversation-panel" aria-labelledby="thread-title"><header className="conversation-heading"><div><p className="eyebrow">SHARED DECISION THREAD</p><h2 id="thread-title">The recovery conversation</h2></div><span>{events.length} events</span></header><div className="conversation-thread" aria-live="polite">{events.map((event,index)=>{const actor=actorFor(event.speaker);const isVerifier=event.speaker==='Verifier';const isSystem=event.speaker==='System';const shipment=shipmentSummary(event.proposal);return <article className={'conversation-entry '+(isVerifier?'verifier-entry ':'')+(event.valid===false?'needs-revision ':'')+(isSystem?'system-entry ':'')} key={`${event.speaker}-${index}`}><div className={'conversation-avatar '+(actor?.accent||'verifier')} aria-hidden="true">{actor?.initial||(isSystem?'·':'✓')}</div><div className="conversation-body"><div className="conversation-meta"><b>{event.speaker}</b>{isVerifier&&<span className={event.valid?'approved':'revision'}>{event.valid?'Approved':'Revision requested'}</span>}{!isVerifier&&!isSystem&&<span className="role-tag">{isMock&&hasRun?'Fallback':'Agent'}</span>}</div><p>{event.message}</p>{shipment&&<div className="shipment-chip"><span>Proposed flow</span>{shipment}</div>}{event.violations?.length>0&&<div className="violation-list">{event.violations.map((violation:any,i:number)=><span key={i}>{violation.constraint}</span>)}</div>}</div></article>;})}</div></section>
      <aside className="outcome-panel" aria-labelledby="outcome-title"><p className="eyebrow">DECISION RECEIPT</p><h2 id="outcome-title">Designed for trust, not theatre.</h2><p className="outcome-copy">Agents surface trade-offs. The verifier checks feasibility. The final trail shows exactly what was proposed, rejected, revised, and executed.</p><dl className="outcome-stats"><div><dt>Participants</dt><dd>4</dd><small>role-specific agents</small></div><div><dt>Revisions</dt><dd>{hasRun?revisions:'—'}</dd><small>{hasRun?'verifier requests':'available in a live run'}</small></div><div><dt>Execution gate</dt><dd>{hasRun?(approved?'Closed':'Review'):'Verifier'}</dd><small>{hasRun?(approved?'safe plan recorded':'awaiting a valid plan'):'always deterministic'}</small></div></dl><div className="outcome-note"><span aria-hidden="true">i</span><p>{hasRun?'This transcript is generated from the completed scenario in this browser session.':'Run a scenario to replace this preview with the current session’s actual negotiation log.'}</p></div></aside>
    </div>
  </section>;
}
