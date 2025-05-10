import nextApi from '@/api/next/instances';
import { queryKeys } from '@/api/query_keys';
import { useQuery } from '@tanstack/react-query';

export const useGetCurrencies = () => {
  return useQuery({
    ...queryKeys.getCurrencies,
    queryFn: async () => {
      return nextApi
        .get<{ code: string; description: string; flag: string }[]>('/api/currency')
        .then(async (res) => res.data);
    }
  });
};
