import React from 'react';
import { ColorAppSegments } from './config';
import { type SegmentedItem, AppSegmented } from '@/components/ui/segmented';
import { useMemoizedFn } from '@/hooks';

const options: SegmentedItem<ColorAppSegments>[] = [
  {
    icon: (
      <div
        className="h-3 w-3 rounded-full"
        style={{
          background: 'linear-gradient(-45deg, #F87B8D, #AA8EFE, #F7B528)'
        }}></div>
    ),
    label: 'Colorful',
    value: ColorAppSegments.Colorful
  },
  {
    icon: (
      <div
        className="h-3 w-3 rounded-full"
        style={{ background: 'linear-gradient(-45deg, #2958E9, #5B9AFA)' }}></div>
    ),
    label: 'Monochrome',
    value: ColorAppSegments.Monochrome
  }
];

export const ColorStyleSegments: React.FC<{
  setSelectedSegment: (value: ColorAppSegments) => void;
  selectedSegment: ColorAppSegments;
}> = React.memo(({ selectedSegment, setSelectedSegment }) => {
  const onChange = useMemoizedFn((value: SegmentedItem<ColorAppSegments>) => {
    setSelectedSegment(value.value);
  });

  return <AppSegmented block options={options} value={selectedSegment} onChange={onChange} />;
});
ColorStyleSegments.displayName = 'PaletteSegments';
