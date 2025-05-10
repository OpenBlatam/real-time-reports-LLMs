import React from 'react';
import { ShareButton } from './ShareButton';
import { ShareMenu } from '../ShareMenu';
import { ShareAssetType } from '@/api/asset_interfaces';
import { getShareAssetConfig } from '../ShareMenu/helpers';
import { useGetCollection } from '@/api/buster_rest/collections';

export const ShareCollectionButton = React.memo(({ collectionId }: { collectionId: string }) => {
  const { data: collectionResponse } = useGetCollection(collectionId, getShareAssetConfig);

  return (
    <ShareMenu
      shareAssetConfig={collectionResponse || null}
      assetId={collectionId}
      assetType={ShareAssetType.COLLECTION}>
      <ShareButton />
    </ShareMenu>
  );
});

ShareCollectionButton.displayName = 'ShareCollectionButton';
