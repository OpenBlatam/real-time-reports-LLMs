import { useMemo } from 'react';
import { listMetrics } from './requests';
import { QueryClient, useQuery, UseQueryOptions } from '@tanstack/react-query';
import { useMemoizedFn } from '@/hooks';
import { metricsQueryKeys } from '@/api/query_keys/metric';
import { RustApiError } from '../errors';
import { isQueryStale } from '@/lib';

export const useGetMetricsList = (
  params: Omit<Parameters<typeof listMetrics>[0], 'page_token' | 'page_size'>,
  options?: Omit<
    UseQueryOptions<Awaited<ReturnType<typeof listMetrics>>, RustApiError>,
    'queryKey' | 'queryFn' | 'initialData'
  >
) => {
  const compiledParams: Parameters<typeof listMetrics>[0] = useMemo(
    () => ({ status: [], ...params, page_token: 0, page_size: 3500 }),
    [params]
  );

  const queryFn = useMemoizedFn(() => listMetrics(compiledParams));

  return useQuery({
    ...metricsQueryKeys.metricsGetList(compiledParams),
    queryFn,
    select: options?.select,
    ...options
  });
};

export const prefetchGetMetricsList = async (
  queryClient: QueryClient,
  params?: Parameters<typeof listMetrics>[0]
) => {
  const options = metricsQueryKeys.metricsGetList(params);
  const isStale = isQueryStale(options, queryClient);
  if (!isStale) return queryClient;

  const lastQueryKey = options.queryKey[options.queryKey.length - 1];
  const compiledParams = lastQueryKey as Parameters<typeof listMetrics>[0];

  await queryClient.prefetchQuery({
    ...options,
    queryFn: async () => {
      return await listMetrics(compiledParams);
    }
  });
  return queryClient;
};
