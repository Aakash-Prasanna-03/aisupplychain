export type Disruption={type:string;start_day:number;duration:number;severity:number};
export type Node={id:string;name:string;inventory:number;capacity:number;production_capacity:number;service_level:number;status:string};
export type State={id?:string;day:number;nodes:Node[];total_cost:number;history:any[];negotiation:any[];verification:any;mock_agents:boolean;agent_model?:string;forecast?:any;meta?:any};
