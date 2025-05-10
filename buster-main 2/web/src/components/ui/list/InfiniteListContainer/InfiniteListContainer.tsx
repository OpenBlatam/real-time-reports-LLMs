import { cn } from '@/lib/classMerge';
import React from 'react';

export const InfiniteListContainer: React.FC<{
  children: React.ReactNode;
  popupNode?: React.ReactNode;
  showContainerBorder?: boolean;
}> = React.memo(({ children, popupNode, showContainerBorder = true }) => {
  return (
    <div className={cn('overflow-hidden', showContainerBorder && 'border-border rounded border')}>
      {children}

      {popupNode && (
        <div className="fixed right-0 bottom-0 left-0 w-full">
          <div className="relative mr-[55px] ml-[220px]">{popupNode}</div>
        </div>
      )}
    </div>
  );
});

InfiniteListContainer.displayName = 'InfiniteListContainer';
