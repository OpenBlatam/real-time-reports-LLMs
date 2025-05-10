import { createTeam, getTeamsList } from './requests';
import { useMutation, useQuery } from '@tanstack/react-query';

export const useCreateTeam = () => {
  return useMutation({
    mutationFn: createTeam
  });
};

export const useGetTeamsList = (params: Parameters<typeof getTeamsList>[0]) => {
  return useQuery({
    queryKey: ['teams'],
    queryFn: () => getTeamsList(params)
  });
};
