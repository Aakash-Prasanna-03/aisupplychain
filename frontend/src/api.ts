import type {Disruption,ExperimentConfig,NetworkImport} from './types';
const BASE='http://localhost:8000/api';
async function request(path:string,init?:RequestInit){const response=await fetch(`${BASE}${path}`,init);if(!response.ok)throw new Error(`API request failed (${response.status})`);return response.json();}
export function create(mode:string,disruption:Disruption,experiment:ExperimentConfig,network?:NetworkImport|null){return request('/simulation',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({mode,disruption,experiment,simulation_days:experiment.simulation_horizon,seed:42,...(network?{network}:{})})});}
export function step(id:string){return request(`/simulation/${id}/step`,{method:'POST'});}
export function negotiate(id:string){return request(`/simulation/${id}/negotiate`,{method:'POST'});}
export function experiment(disruption:Disruption,experiment:ExperimentConfig,network?:NetworkImport|null){return request('/experiment',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({mode:'verified',disruption,experiment,simulation_days:experiment.simulation_horizon,seed:42,...(network?{network}:{})})});}
export function interpretScenario(prompt:string,experiment:ExperimentConfig){return request('/scenario/interpret',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({prompt,experiment})});}
export function runFullSimulation(disruption:Disruption,experiment:ExperimentConfig,network?:NetworkImport|null){return request('/simulation/run',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({mode:'verified',disruption,experiment,simulation_days:experiment.simulation_horizon,seed:42,...(network?{network}:{})})});}
export function getArdnConfig(){return request('/ardn/config');}
export function updateArdnConfig(tuning:Record<string,unknown>){return request('/ardn/config',{method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify(tuning)});}
export async function downloadRunReport(id:string){const response=await fetch(`${BASE}/simulation/${id}/report`);if(!response.ok)throw new Error(`Report request failed (${response.status})`);const blob=await response.blob();const url=URL.createObjectURL(blob);const link=document.createElement('a');link.href=url;link.download=`recovery-report-${id}.pdf`;document.body.appendChild(link);link.click();link.remove();URL.revokeObjectURL(url);}

