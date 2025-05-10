import { createInstance } from '../createInstance';
import { BASE_URL } from './config';

const mainApi = createInstance(BASE_URL);
export default mainApi;
export { mainApi };
