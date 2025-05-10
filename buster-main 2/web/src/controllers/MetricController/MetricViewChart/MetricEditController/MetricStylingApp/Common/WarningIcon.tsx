import { TriangleWarning } from '@/components/ui/icons';
import React from 'react';
import { cn } from '@/lib/classMerge';
import { Popover } from '@/components/ui/popover/Popover';

export const WarningIcon: React.FC<{
  rowCountThreshold?: number;
  rowCount: number;

  warningText?: string;
}> = React.memo(
  ({
    rowCount,
    rowCountThreshold = 25,
    warningText = 'Data labels will be hidden if there are too many.'
  }) => {
    if (rowCount <= rowCountThreshold) {
      return null;
    }
    return (
      <Popover
        side="left"
        align="center"
        size="none"
        content={<div className="max-w-[170px] p-2">{warningText}</div>}>
        <div
          className={cn(
            'text-text-tertiary hover:text-text-secondary flex h-full cursor-pointer items-center'
          )}>
          <TriangleWarning />
        </div>
      </Popover>
    );
  }
);
WarningIcon.displayName = 'WarningIcon';
