import React from 'react';
import { Title, Text } from '@/components/ui/typography';
import { Avatar } from '@/components/ui/avatar';
import type { OrganizationUser } from '@/api/asset_interfaces';

export const UserHeader = React.memo(({ user }: { user: OrganizationUser }) => {
  return (
    <div className="flex justify-between">
      <UserInfo user={user} />
    </div>
  );
});

UserHeader.displayName = 'UserHeader';

const UserInfo: React.FC<{ user: OrganizationUser }> = ({ user }) => {
  return (
    <div className="flex items-center space-x-3">
      <Avatar size={32} fallbackClassName="text-base" name={user.name} />
      <div className="flex flex-col space-y-0.5">
        <Title as="h4">{user.name}</Title>
        <Text size="sm" variant="secondary">
          {user.email}
        </Text>
      </div>
    </div>
  );
};
