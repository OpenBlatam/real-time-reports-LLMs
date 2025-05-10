import { RustApiError } from '@/api/buster_rest/errors';
import { QueryClient, queryOptions } from '@tanstack/react-query';

export const isQueryStale = (
  options: ReturnType<typeof queryOptions<any, RustApiError, any>>,
  queryClient: QueryClient
): boolean => {
  const queryState = queryClient.getQueryState(options.queryKey);
  const updatedAt = queryState?.dataUpdatedAt;
  const staleTime =
    (options.staleTime as number) ||
    (queryClient.getDefaultOptions().queries?.staleTime as number) ||
    0;
  const isStale = updatedAt ? Date.now() - updatedAt > staleTime : true;

  return isStale;
};
