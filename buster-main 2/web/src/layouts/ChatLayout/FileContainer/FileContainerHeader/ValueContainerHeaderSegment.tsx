import React from 'react';
import { FileContainerSegmentProps } from './interfaces';

export const ValueContainerHeaderSegment: React.FC<FileContainerSegmentProps> = React.memo(() => {
  return <div>Value Container Header</div>;
});

ValueContainerHeaderSegment.displayName = 'ValueContainerHeaderSegment';
