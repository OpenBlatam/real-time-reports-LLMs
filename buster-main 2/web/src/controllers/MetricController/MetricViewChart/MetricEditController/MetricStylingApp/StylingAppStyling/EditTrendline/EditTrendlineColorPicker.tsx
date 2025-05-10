import { ColorPicker } from '@/components/ui/color-picker';
import React from 'react';
import { LabelAndInput } from '../../Common';
import { LoopTrendline } from './EditTrendline';
import { useMemoizedFn } from '@/hooks';

export const TrendlineColorPicker = React.memo(
  ({
    trend,
    onUpdateExisitingTrendline
  }: {
    trend: LoopTrendline;
    onUpdateExisitingTrendline: (trend: LoopTrendline) => void;
  }) => {
    const onChangeComplete = useMemoizedFn((color: string) => {
      const hexColor = color;
      onUpdateExisitingTrendline({ ...trend, trendLineColor: hexColor });
    });

    return (
      <LabelAndInput label="Color">
        <div className="flex w-full items-center justify-end">
          <ColorPicker
            size="small"
            value={trend.trendLineColor || 'black'}
            onChangeComplete={onChangeComplete}
          />
        </div>
      </LabelAndInput>
    );
  }
);
TrendlineColorPicker.displayName = 'TrendlineColorPicker';
