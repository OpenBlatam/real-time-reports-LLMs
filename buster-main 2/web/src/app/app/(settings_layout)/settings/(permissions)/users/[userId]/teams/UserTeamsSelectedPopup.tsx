import { useUpdateUserTeams } from '@/api/buster_rest';
import type { TeamRole } from '@/api/asset_interfaces';
import { BusterListSelectedOptionPopupContainer } from '@/components/ui/list';
import { useMemoizedFn } from '@/hooks';
import React from 'react';
import { PermissionAssignTeamRoleButton } from '@/components/features/PermissionComponents';

export const UserTeamsSelectedPopup: React.FC<{
  selectedRowKeys: string[];
  onSelectChange: (selectedRowKeys: string[]) => void;
  userId: string;
}> = React.memo(({ selectedRowKeys, onSelectChange, userId }) => {
  const { mutateAsync: updateUserTeams } = useUpdateUserTeams({
    userId: userId
  });

  const onRoleChange = useMemoizedFn(async (role: TeamRole) => {
    await updateUserTeams(
      selectedRowKeys.map((id) => ({
        id,
        role
      }))
    );
    onSelectChange([]);
  });

  return (
    <BusterListSelectedOptionPopupContainer
      selectedRowKeys={selectedRowKeys}
      onSelectChange={onSelectChange}
      buttons={[<PermissionAssignTeamRoleButton key="assign" onRoleChange={onRoleChange} />]}
    />
  );
});

UserTeamsSelectedPopup.displayName = 'UserTeamsSelectedPopup';
