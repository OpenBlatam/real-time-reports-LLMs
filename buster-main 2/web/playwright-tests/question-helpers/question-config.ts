import { ChartType } from '@/api/asset_interfaces';

type QuestionConfig = {
  version: number;
  selectedChartType: ChartType;
  id: string;
  description: string;
};

export const metricConfig: Record<string, QuestionConfig> = {
  'Total Unique Products Sold': {
    version: 1,
    selectedChartType: ChartType.Metric,
    id: '5316b39f-54ca-59b4-9102-9f581e1fa680',
    description: 'Basic metric chart'
  },
  'Yearly Sales Revenue - Signature Cycles Products (Last 3 Years + YTD)': {
    version: 1,
    selectedChartType: ChartType.Bar,
    id: '45c17750-2b61-5683-ba8d-ff6c6fefacee',
    description: 'Bar chart with 1 dataset and 2 data points'
  },
  'Top 10 Most Active Vendors by Purchase Order Count (Last 3 Years)': {
    version: 1,
    selectedChartType: ChartType.Bar,
    id: 'b401778a-bd93-53f4-b884-236aaf62c9a8',
    description: 'Bar chart with 1 dataset and 2 data points it is horizontal'
  },
  'Yearly Sales Revenue by Product - Signature Cycles (2022-Present)': {
    version: 1,
    selectedChartType: ChartType.Bar,
    id: '2b569e92-229b-5cad-b312-b09c751c544d',
    description: 'Bar chart with 2 datasets and 2 data points it is vertical'
  }
};
