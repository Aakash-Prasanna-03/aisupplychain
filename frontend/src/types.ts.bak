export type DisruptionEffect={type:string;target?:string;source?:string;destination?:string;magnitude:number;description:string};
export type Disruption={description:string;affected_nodes:string[];affected_routes:any[];effects:DisruptionEffect[];start_day:number;duration:number;severity:number};
export type ExperimentConfig={severity:number|null;disruption_duration:number|null;simulation_horizon:number;max_negotiation_rounds:number;recovery_aggressiveness:'conservative'|'balanced'|'aggressive'};
export type Node={id:string;name:string;inventory:number;capacity:number;production_capacity:number;service_level:number;status:string};
export type State={id?:string;day:number;nodes:Node[];total_cost:number;history:any[];negotiation:any[];verification:any;mock_agents:boolean;agent_model?:string;forecast?:any;meta?:any;scenario?:Disruption;experiment?:ExperimentConfig};
