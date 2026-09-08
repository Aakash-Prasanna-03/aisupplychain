import type {Disruption} from './types';
const BASE='http://localhost:8000/api';
async function request(path:string,init?:RequestInit){const response=await fetch(`${BASE}${path}`,init);if(!response.ok)throw new Error(`API request failed (${response.status})`);return response.json();}
export function create(mode:string,disruption:Disruption){return request('/simulation',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({mode,disruption,simulation_days:12,seed:42})});}
export function step(id:string){return request(`/simulation/${id}/step`,{method:'POST'});}
export function negotiate(id:string){return request(`/simulation/${id}/negotiate`,{method:'POST'});}
export function experiment(disruption:Disruption){return request('/experiment',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({mode:'verified',disruption,simulation_days:12,seed:42})});}
