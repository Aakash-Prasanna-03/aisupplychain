import type {Disruption} from './types'; const BASE='http://localhost:8000/api';
export async function create(mode:string,disruption:Disruption){return (await fetch(`${BASE}/simulation`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({mode,disruption,simulation_days:12,seed:42})})).json()}
export async function negotiate(id:string){return (await fetch(`${BASE}/simulation/${id}/negotiate`,{method:'POST'})).json()}
export async function experiment(disruption:Disruption){return (await fetch(`${BASE}/experiment`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({mode:'verified',disruption,simulation_days:12,seed:42})})).json()}
