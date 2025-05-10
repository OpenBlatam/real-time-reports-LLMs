'use client';

import { BackButton } from '@/components/ui/buttons';
import { type ISidebarGroup, Sidebar } from '@/components/ui/sidebar';
import { CircleUser, LockCircle, ApartmentBuilding } from '@/components/ui/icons';
import { BusterRoutes, createBusterRoute } from '@/routes';
import React, { useMemo } from 'react';
import { useUserConfigContextSelector } from '@/context/Users';
import { useAppLayoutContextSelector } from '@/context/BusterAppLayout';
import { SidebarUserFooter } from './SidebarUserFooter/SidebarUserFooter';

const accountItems: ISidebarGroup = {
  label: 'Account',
  variant: 'icon',
  id: 'account',
  icon: <CircleUser />,
  items: [
    {
      label: 'Profile',
      route: createBusterRoute({ route: BusterRoutes.SETTINGS_PROFILE }),
      id: createBusterRoute({ route: BusterRoutes.SETTINGS_PROFILE })
    }
  ]
};

const workspaceItems: ISidebarGroup = {
  label: 'Workspace',
  variant: 'icon',
  id: 'workspace',
  icon: <ApartmentBuilding />,
  items: [
    {
      label: 'API Keys',
      route: createBusterRoute({ route: BusterRoutes.SETTINGS_API_KEYS }),
      id: createBusterRoute({ route: BusterRoutes.SETTINGS_API_KEYS })
    },
    {
      label: 'Data Sources',
      route: createBusterRoute({ route: BusterRoutes.SETTINGS_DATASOURCES }),
      id: createBusterRoute({ route: BusterRoutes.SETTINGS_DATASOURCES })
    }
  ]
};

const permissionAndSecurityItems: ISidebarGroup = {
  label: 'Permission & Security',
  variant: 'icon',
  id: 'permission-and-security',
  icon: <LockCircle />,
  items: [
    {
      label: 'Users',
      route: createBusterRoute({ route: BusterRoutes.SETTINGS_USERS }),
      id: createBusterRoute({ route: BusterRoutes.SETTINGS_USERS })
    },
    {
      label: 'Dataset groups',
      route: createBusterRoute({ route: BusterRoutes.SETTINGS_DATASET_GROUPS }),
      id: createBusterRoute({ route: BusterRoutes.SETTINGS_DATASET_GROUPS })
    },
    {
      label: 'Permission groups',
      route: createBusterRoute({ route: BusterRoutes.SETTINGS_PERMISSION_GROUPS }),
      id: createBusterRoute({ route: BusterRoutes.SETTINGS_PERMISSION_GROUPS })
    }
  ]
};

export const SidebarSettings: React.FC<{}> = React.memo(({}) => {
  const isAdmin = useUserConfigContextSelector((x) => x.isAdmin);
  const currentParentRoute = useAppLayoutContextSelector((x) => x.currentParentRoute);

  const content = useMemo(() => {
    const items = [accountItems];
    if (isAdmin) {
      items.push(workspaceItems);
      items.push(permissionAndSecurityItems);
    }
    return items;
  }, [isAdmin]);

  return (
    <Sidebar
      content={content}
      header={useMemo(
        () => (
          <SidebarSettingsHeader />
        ),
        []
      )}
      activeItem={currentParentRoute}
      footer={useMemo(
        () => (
          <SidebarUserFooter />
        ),
        []
      )}
    />
  );
});

SidebarSettings.displayName = 'SidebarSettings';

const SidebarSettingsHeader = React.memo(() => {
  return (
    <div>
      <BackButton linkUrl={createBusterRoute({ route: BusterRoutes.APP_HOME })} text="Settings" />
    </div>
  );
});

SidebarSettingsHeader.displayName = 'SidebarSettingsHeader';
