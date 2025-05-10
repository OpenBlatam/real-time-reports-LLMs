import { getNow, isDateAfter, isDateBefore, isDateSame } from '@/lib/date';
import { useMemo } from 'react';

type ListItem = {
  id: string;
  last_edited: string;
};

const createChatRecord = <T extends ListItem>(
  data: T[]
): {
  TODAY: T[];
  YESTERDAY: T[];
  LAST_WEEK: T[];
  ALL_OTHERS: T[];
} => {
  const today = getNow();
  const TODAY = data.filter((d) =>
    isDateSame({
      date: d.last_edited,
      compareDate: today,
      interval: 'day'
    })
  );
  const YESTERDAY = data.filter((d) =>
    isDateSame({
      date: d.last_edited,
      compareDate: today.subtract(1, 'day'),
      interval: 'day'
    })
  );
  const LAST_WEEK = data.filter(
    (d) =>
      isDateBefore({
        date: d.last_edited,
        compareDate: today.subtract(2, 'day').startOf('day'),
        interval: 'day'
      }) &&
      isDateAfter({
        date: d.last_edited,
        compareDate: today.subtract(8, 'day').startOf('day'),
        interval: 'day'
      })
  );
  const ALL_OTHERS = data.filter(
    (d) => !TODAY.includes(d) && !YESTERDAY.includes(d) && !LAST_WEEK.includes(d)
  );

  return {
    TODAY,
    YESTERDAY,
    LAST_WEEK,
    ALL_OTHERS
  };
};

export const useCreateListByDate = <T extends ListItem>({ data }: { data: T[] }) => {
  const listRecord = useMemo(() => createChatRecord(data), [data]);

  return listRecord;
};
