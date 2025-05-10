import { renderHook } from '@testing-library/react';
import { useDataTrendlineOptions } from './useDataTrendlineOptions';
import { ChartType, type ChartEncodes, type Trendline } from '@/api/asset_interfaces/metric/charts';
import { DatasetOptionsWithTicks } from '../interfaces';
import { TrendlineDataset } from './trendlineDataset.types';
import { DEFAULT_COLUMN_LABEL_FORMAT } from '@/api/asset_interfaces/metric';
import { IColumnLabelFormat } from '@/api/asset_interfaces/metric/charts';

// Mock the modules
jest.mock('./canSupportTrendline', () => ({
  canSupportTrendlineRecord: {
    max: jest.fn((columnLabelFormats, trendline) => {
      return trendline.columnId !== 'non-existent-id';
    }),
    min: jest.fn((columnLabelFormats, trendline) => {
      return trendline.columnId !== 'non-existent-id';
    }),
    median: jest.fn((columnLabelFormats, trendline) => {
      return trendline.columnId !== 'non-existent-id';
    }),
    linear_regression: jest.fn((columnLabelFormats, trendline) => {
      return trendline.columnId !== 'non-existent-id';
    }),
    exponential_regression: jest.fn((columnLabelFormats, trendline) => {
      return trendline.columnId !== 'non-existent-id';
    }),
    logarithmic_regression: jest.fn((columnLabelFormats, trendline) => {
      return trendline.columnId !== 'non-existent-id';
    })
  }
}));

jest.mock('./trendlineDatasetCreator', () => ({
  trendlineDatasetCreator: {
    max: jest.fn((trendline, datasetOptions, columnLabelFormats) => {
      if (trendline.columnId === 'non-existent-id') {
        throw new Error('Column not found');
      }
      return [
        {
          id: 'test-column-id_🫷🥸🫸trend_max',
          type: 'max',
          columnId: 'test-column-id',
          data: [25],
          dataKey: 'test-column-id',
          axisType: 'y',
          label: [{ key: 'value', value: 25 }],
          tooltipData: [[{ key: 'value', value: 25 }]],
          show: true,
          showTrendlineLabel: true,
          trendlineLabel: 'Test Trendline'
        }
      ];
    }),
    min: jest.fn((trendline, datasetOptions, columnLabelFormats) => [
      {
        id: 'test-column-id_🫷🥸🫸trend_min',
        type: 'min',
        columnId: 'test-column-id',
        data: [5],
        dataKey: 'test-column-id',
        axisType: 'y',
        label: [{ key: 'value', value: 5 }],
        tooltipData: [[{ key: 'value', value: 5 }]],
        show: true,
        showTrendlineLabel: true,
        trendlineLabel: 'Test Trendline'
      }
    ])
  }
}));

describe('useDataTrendlineOptions', () => {
  const mockDatasetOptions: DatasetOptionsWithTicks = {
    datasets: [
      {
        id: 'test-column-id',
        data: [5, 10, 15, 20, 25],
        label: [{ key: 'test-label', value: 'Test Label' }],
        dataKey: 'test-column-id',
        axisType: 'y',
        tooltipData: [[{ key: 'test-tooltip', value: 'Test Tooltip' }]]
      }
    ],
    ticks: [['1'], ['2'], ['3'], ['4'], ['5']],
    ticksKey: [{ key: 'test-column-id', value: '' }]
  };

  const mockColumnLabelFormats: Record<string, IColumnLabelFormat> = {
    'test-column-id': {
      ...DEFAULT_COLUMN_LABEL_FORMAT,
      columnType: 'number',
      style: 'number'
    }
  };

  const mockSelectedAxis: ChartEncodes = {
    x: ['test-column-id'],
    y: ['test-column-id']
  };

  const defaultProps = {
    datasetOptions: mockDatasetOptions,
    trendlines: [] as Trendline[],
    selectedChartType: ChartType.Line,
    selectedAxis: mockSelectedAxis,
    columnLabelFormats: mockColumnLabelFormats
  };

  const createMockTrendline = (type: Trendline['type']): Trendline => ({
    type,
    columnId: 'test-column-id',
    show: true,
    showTrendlineLabel: true,
    trendlineLabel: 'Test Trendline'
  });

  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('should return empty array when no trendlines are provided', () => {
    const { result } = renderHook(() => useDataTrendlineOptions(defaultProps));
    expect(result.current).toEqual([]);
  });

  it('should return empty array when datasetOptions is undefined', () => {
    const props = {
      ...defaultProps,
      datasetOptions: undefined,
      trendlines: [createMockTrendline('max')]
    };
    const { result } = renderHook(() => useDataTrendlineOptions(props));
    expect(result.current).toEqual([]);
  });

  it('should return empty array when datasets array is empty', () => {
    const props = {
      ...defaultProps,
      datasetOptions: { ...mockDatasetOptions, datasets: [] },
      trendlines: [createMockTrendline('max')]
    };
    const { result } = renderHook(() => useDataTrendlineOptions(props));
    expect(result.current).toEqual([]);
  });

  it('should support trendlines for line charts', () => {
    const props = {
      ...defaultProps,
      trendlines: [createMockTrendline('max')]
    };
    const { result } = renderHook(() => useDataTrendlineOptions(props));
    expect(result.current.length).toBeGreaterThan(0);
  });

  it('should support trendlines for bar charts', () => {
    const props = {
      ...defaultProps,
      selectedChartType: ChartType.Bar,
      trendlines: [createMockTrendline('max')]
    };
    const { result } = renderHook(() => useDataTrendlineOptions(props));
    expect(result.current.length).toBeGreaterThan(0);
  });

  it('should support trendlines for scatter charts', () => {
    const props = {
      ...defaultProps,
      selectedChartType: ChartType.Scatter,
      trendlines: [createMockTrendline('max')]
    };
    const { result } = renderHook(() => useDataTrendlineOptions(props));
    expect(result.current.length).toBeGreaterThan(0);
  });

  it('should support trendlines for combo charts', () => {
    const props = {
      ...defaultProps,
      selectedChartType: ChartType.Combo,
      trendlines: [createMockTrendline('max')]
    };
    const { result } = renderHook(() => useDataTrendlineOptions(props));
    expect(result.current.length).toBeGreaterThan(0);
  });

  it('should not support trendlines for unsupported chart types', () => {
    const props = {
      ...defaultProps,
      selectedChartType: ChartType.Pie,
      trendlines: [createMockTrendline('max')]
    };
    const { result } = renderHook(() => useDataTrendlineOptions(props));
    expect(result.current).toEqual([]);
  });

  it('should handle multiple trendlines', () => {
    const props = {
      ...defaultProps,
      trendlines: [createMockTrendline('max'), createMockTrendline('min')]
    };
    const { result } = renderHook(() => useDataTrendlineOptions(props));
    expect(result.current.length).toBeGreaterThan(1);
  });

  it('should handle unsupported trendline types gracefully', () => {
    const props = {
      ...defaultProps,
      trendlines: [{ ...createMockTrendline('max'), type: 'unsupported' as any }]
    };
    const { result } = renderHook(() => useDataTrendlineOptions(props));
    expect(result.current).toEqual([]);
  });

  it('should handle errors in trendline creation gracefully', () => {
    const props = {
      ...defaultProps,
      trendlines: [{ ...createMockTrendline('max'), columnId: 'non-existent-id' }]
    };
    const { result } = renderHook(() => useDataTrendlineOptions(props));
    expect(result.current).toEqual([]);
  });

  it('should memoize results based on dependencies', () => {
    const props = {
      ...defaultProps,
      trendlines: [createMockTrendline('max')]
    };
    const { result, rerender } = renderHook((props) => useDataTrendlineOptions(props), {
      initialProps: props
    });
    const firstResult = result.current;

    // Rerender with same props
    rerender(props);
    expect(result.current).toBe(firstResult);

    // Rerender with different trendlines
    const newProps = {
      ...props,
      trendlines: [createMockTrendline('min')]
    };
    rerender(newProps);
    const secondResult = result.current;
    expect(secondResult).not.toBe(firstResult);
    expect(secondResult[0].id).toBe('test-column-id_🫷🥸🫸trend_min');
  });
});
