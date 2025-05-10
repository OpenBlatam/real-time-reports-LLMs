import React from 'react';
import { ISidebarGroup, ISidebarList, SidebarProps } from './interfaces';
import { SidebarCollapsible } from './SidebarCollapsible';
import { SidebarItem } from './SidebarItem';

export const Sidebar: React.FC<SidebarProps> = React.memo(
  ({ header, content, footer, activeItem }) => {
    return (
      <div className="flex h-full flex-col overflow-hidden px-3.5 pt-4.5">
        <div className="flex flex-col space-y-4.5 overflow-hidden">
          <div className="mb-5"> {header}</div>
          <div className="flex flex-grow flex-col space-y-4.5 overflow-y-auto pb-3">
            {content.map((item) => (
              <ContentSelector key={item.id} content={item} activeItem={activeItem} />
            ))}
          </div>
        </div>
        {footer && <div className="mt-auto mb-2 overflow-hidden pt-5">{footer}</div>}
      </div>
    );
  }
);

Sidebar.displayName = 'Sidebar';

const ContentSelector: React.FC<{
  content: SidebarProps['content'][number];
  activeItem: SidebarProps['activeItem'];
}> = React.memo(({ content, activeItem }) => {
  if (isSidebarGroup(content)) {
    return <SidebarCollapsible {...content} activeItem={activeItem} />;
  }

  return <SidebarList items={content.items} activeItem={activeItem} />;
});
ContentSelector.displayName = 'ContentSelector';

const SidebarList: React.FC<{
  items: ISidebarList['items'];
  activeItem: SidebarProps['activeItem'];
}> = ({ items, activeItem }) => {
  return (
    <div className="flex flex-col space-y-0.5">
      {items.map((item) => (
        <SidebarItem key={item.id} {...item} active={activeItem === item.id || item.active} />
      ))}
    </div>
  );
};

const isSidebarGroup = (content: SidebarProps['content'][number]): content is ISidebarGroup => {
  return 'label' in content && content.label !== undefined;
};
