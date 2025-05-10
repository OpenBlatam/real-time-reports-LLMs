import debounce from 'lodash/debounce';
import memoize from 'lodash/memoize';
import wrap from 'lodash/wrap';

export const timeout = (time = 1000) => {
  return new Promise((resolve) => {
    setTimeout(() => {
      resolve(true);
    }, time);
  });
};

export const createDebounceByParams = (
  func: (...args: any) => any,
  resolver: any,
  d: { delay: number }
) => {
  /*const hocSaveDebounced = createDebounceByParams(
  save,
  (d: any) => {
    return d.id; //set resolver to cache by id
  },
  { delay: 500 }
);
*/
  return wrap(
    memoize(() => debounce(func, d.delay), resolver),
    //@ts-ignore
    (getMemoizedFunc, obj: any) => getMemoizedFunc(obj)(obj)
  );
};
