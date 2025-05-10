import truncate from 'lodash/truncate';

export const truncateWithEllipsis = (text: string, maxLength: number = 52) =>
  text.length > maxLength ? truncate(text, { length: maxLength }) : text;
