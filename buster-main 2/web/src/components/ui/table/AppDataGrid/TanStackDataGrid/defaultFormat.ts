import { makeHumanReadble } from '@/lib/text';

export const defaultHeaderFormat = (v: any) => makeHumanReadble(v);
export const defaultCellFormat = (v: any) => v;
