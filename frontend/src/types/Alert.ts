export interface Alert {
  id: string;
  msg: string;
  type: string;
}

export interface AlertState {
  alerts: Alert[];
}