import React from 'react';
import { Title, Paragraph } from '@/components/ui/typography';
import { Button } from '@/components/ui/buttons';
import { Plus } from '@/components/ui/icons';
import Link from 'next/link';

export const ListEmptyStateWithButton: React.FC<{
  isAdmin?: boolean;
  title: string;
  description: string;
  onClick?: () => void;
  buttonText: string;
  loading?: boolean;
  linkButton?: string;
}> = React.memo(
  ({ isAdmin = true, linkButton, title, buttonText, description, onClick, loading = false }) => {
    return (
      <div className="flex h-full w-full flex-col">
        <div
          className="flex h-full w-full flex-col items-center justify-start space-y-5 text-center"
          style={{
            marginTop: '25vh'
          }}>
          <div className="flex w-[350px] flex-col justify-center space-y-3">
            <Title as="h4" className="leading-1.3 text-center [text-wrap:balance]">
              {title}
            </Title>

            <Paragraph variant="secondary">{description}</Paragraph>
          </div>

          {isAdmin && (
            <ButtonWrapper href={linkButton}>
              <Button variant="default" prefix={<Plus />} loading={loading} onClick={onClick}>
                {buttonText}
              </Button>
            </ButtonWrapper>
          )}
        </div>
      </div>
    );
  }
);

const ButtonWrapper: React.FC<{
  children: React.ReactNode;
  href?: string;
}> = ({ children, href }) => {
  if (!href) return <>{children}</>;
  return <Link href={href}>{children}</Link>;
};

ListEmptyStateWithButton.displayName = 'ListEmptyStateWithButton';
