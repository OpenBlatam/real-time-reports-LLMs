import mainApi from '@/api/buster_rest/instances';
import type { BusterApiKeyListItem } from '@/api/asset_interfaces/api_keys';

export const getApiKeys = async () => {
  return mainApi.get<{ api_keys: BusterApiKeyListItem[] }>(`/api_keys`).then((res) => res.data);
};

export const createApiKey = async (name: string) => {
  return mainApi.post<{ api_key: string }>(`/api_keys`, { name }).then((res) => res.data);
};

export const deleteApiKey = async (id: string) => {
  return mainApi.delete(`/api_keys/${id}`).then((res) => res.data);
};

export const getApiKey = async (id: string) => {
  return mainApi.get<BusterApiKeyListItem>(`/api_keys/${id}`).then((res) => res.data);
};
