//Goal line is a line that is drawn on the chart to represent a goal.
export interface GoalLine {
  show: boolean; //OPTIONAL: default is false. this should only be used if the user explicitly requests a goal line
  value: number; //OPTIONAL: default is null. it should remain null until the user specifies what the goal line value should be.
  showGoalLineLabel: boolean; //OPTIONAL: default is true.
  goalLineLabel: string | null; //OPTIONAL: if showGoalLineLabel is true, this will be the label. default is "Goal".
  goalLineColor?: string | null; //OPTIONAL: default is #000000
}

export interface Trendline {
  show: boolean; //OPTIONAL: default is true. this should only be used if the user explicitly requests a trendline
  showTrendlineLabel: boolean; //OPTIONAL: default is true
  trendlineLabel: string | null; //OPTIONAL: if showTrendlineLabel is true, this will be the label. default is "Slope".
  type:
    | 'average'
    | 'linear_regression'
    | 'logarithmic_regression'
    | 'exponential_regression'
    | 'polynomial_regression'
    | 'min'
    | 'max'
    | 'median'; //default is linear trend
  trendLineColor?: string | null; //OPTIONAL: default is #000000
  columnId: string;
  id?: string;
}
