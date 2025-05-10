import React from 'react';
import { StylingAppColorsTab } from './config';
import { type SegmentedItem, AppSegmented } from '@/components/ui/segmented';
import { Palette, Cards, Paintbrush } from '@/components/ui/icons';
import { useMemoizedFn } from '@/hooks';

const options: SegmentedItem<StylingAppColorsTab>[] = [
  { label: 'Colors', icon: <Palette />, value: StylingAppColorsTab.Colors },
  { label: 'Palettes', icon: <Cards />, value: StylingAppColorsTab.Palettes },
  { label: 'Custom', icon: <Paintbrush />, value: StylingAppColorsTab.Custom }
];

export const SelectColorApp: React.FC<{
  selectedTab: StylingAppColorsTab;
  onChange: (value: StylingAppColorsTab) => void;
}> = React.memo(({ selectedTab, onChange }) => {
  const onChangePreflight = useMemoizedFn((value: SegmentedItem<StylingAppColorsTab>) => {
    onChange(value.value);
  });

  return (
    <AppSegmented
      size="large"
      block
      options={options}
      value={selectedTab}
      onChange={onChangePreflight}
    />
  );
});
SelectColorApp.displayName = 'SelectColorApp';
