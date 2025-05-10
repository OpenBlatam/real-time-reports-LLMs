import isString from 'lodash/isString';

export const rustErrorHandler = (errors: any = {}): RustApiError => {
  const data = errors?.response?.data;
  const status = errors?.status;

  if (data && isString(data)) {
    return { message: String(data), status };
  }

  if (data && data?.message) {
    return { message: String(data.message), status };
  }

  if (data && data?.detail) {
    if (typeof data.detail === 'string') {
      return { message: String(data.detail), status };
    }

    if (data.detail?.[0]) {
      return { message: String(data.detail[0].msg), status };
    }
    return { message: String(data.detail), status };
  }
  if (errors?.message) {
    return { message: String(errors.message), status };
  }

  if (typeof errors === 'string') {
    return { message: String(errors), status };
  }

  return {};
};

export interface RustApiError {
  message?: string;
  status?: number;
}
