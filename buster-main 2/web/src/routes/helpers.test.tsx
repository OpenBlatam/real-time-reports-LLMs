import { BusterRoutes } from './busterRoutes';
import { pathNameToRoute } from './helpers';

describe('pathNameToRoute', () => {
  test('should return ROOT for unmatched route', () => {
    const result = pathNameToRoute('/invalid/path', {});
    expect(result).toBe(BusterRoutes.ROOT);
  });

  test('should return parent route for chat ID route', () => {
    const pathName = '/app/chats/123';
    const params = { chatId: '123' };
    const result = pathNameToRoute(pathName, params);
    expect(result).toBe(BusterRoutes.APP_CHAT_ID);
  });

  test('should return route as is when no mapping exists', () => {
    const pathName = '/app/home';
    const result = pathNameToRoute(pathName, {});
    expect(result).toBe(BusterRoutes.APP_HOME);
  });

  test('/app/metrics/70d34bca-79a2-5532-9d51-8fd2f2525f81/file', () => {
    const pathName = '/app/metrics/70d34bca-79a2-5532-9d51-8fd2f2525f81/file';
    const result = pathNameToRoute(pathName, {
      metricId: '70d34bca-79a2-5532-9d51-8fd2f2525f81'
    });
    expect(result).toBe(BusterRoutes.APP_METRIC_ID_FILE__HIDDEN);
  });
});
