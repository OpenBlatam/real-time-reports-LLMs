import React from 'react';
import { ChatHeader } from './ChatHeader';
import { ChatContent } from './ChatContent';
import { AppPageLayout } from '@/components/ui/layouts';

export const ChatContainer = React.memo(({ mounted }: { mounted?: boolean }) => {
  return (
    <AppPageLayout
      header={<ChatHeader />}
      headerBorderVariant="ghost"
      scrollable
      className="chat-container-content flex h-full w-full min-w-[295px] flex-col">
      <ChatContent />
    </AppPageLayout>
  );
});

ChatContainer.displayName = 'ChatContainer';
