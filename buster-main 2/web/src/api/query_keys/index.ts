import { chatQueryKeys } from './chat';
import { collectionQueryKeys } from './collection';
import { userQueryKeys } from './users';
import { dashboardQueryKeys } from './dashboard';
import { metricsQueryKeys } from './metric';
import { searchQueryKeys } from './search';
import { termsQueryKeys } from './terms';
import { datasourceQueryKeys } from './datasources';
import { datasetGroupQueryKeys } from './dataset_groups';
import { datasetQueryKeys } from './datasets';
import { permissionGroupQueryKeys } from './permission_groups';
import { nextjsQueryKeys } from './nextjs';

export const queryKeys = {
  ...datasetQueryKeys,
  ...chatQueryKeys,
  ...collectionQueryKeys,
  ...userQueryKeys,
  ...dashboardQueryKeys,
  ...metricsQueryKeys,
  ...searchQueryKeys,
  ...termsQueryKeys,
  ...datasourceQueryKeys,
  ...datasetGroupQueryKeys,
  ...permissionGroupQueryKeys,
  ...nextjsQueryKeys
};
