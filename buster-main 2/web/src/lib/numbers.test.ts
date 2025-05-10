import { formatNumber } from './numbers';

describe('formatNumber', () => {
  test('handles null and undefined values', () => {
    expect(formatNumber(null)).toBe('');
    expect(formatNumber(undefined)).toBe('');
  });

  test('formats basic numbers', () => {
    expect(formatNumber(1234.5678)).toBe('1,234.568');
    expect(formatNumber(1000)).toBe('1,000');
    expect(formatNumber(0)).toBe('0');
  });

  test('respects decimal places settings', () => {
    expect(formatNumber(1234.5678, { minDecimals: 3 })).toBe('1,234.568');
    expect(formatNumber(1234.5, { minDecimals: 3 })).toBe('1,234.500');
    expect(formatNumber(1234.5678, { maximumDecimals: 1 })).toBe('1,234.6');
  });

  test('handles compact notation', () => {
    expect(formatNumber(1_000_000, { compact: true })).toBe('1M');
    expect(formatNumber(1_500_000, { compact: true })).toBe('1.5M');
    expect(formatNumber(1_000, { compact: true })).toBe('1K');
  });

  test('handles different locales', () => {
    expect(formatNumber(1234.56, { locale: 'de-DE' })).toBe('1.234,56');
    expect(formatNumber(1234.56, { locale: 'fr-FR' })).toBe('1 234,56');
  });

  test('handles currency formatting', () => {
    expect(formatNumber(1234.56, { currency: 'USD' })).toBe('$1,234.56');
    expect(formatNumber(1234.56, { currency: 'EUR' })).toBe('€1,234.56');
  });

  test('handles string inputs', () => {
    expect(formatNumber('1234.56')).toBe('1,234.56');
    expect(formatNumber('1000')).toBe('1,000');
  });

  test('handles non-numeric strings', () => {
    expect(formatNumber('not a number')).toBe('not a number');
    expect(formatNumber('')).toBe('');
  });

  test('handles grouping options', () => {
    expect(formatNumber(1234567, { useGrouping: false })).toBe('1234567');
    expect(formatNumber(1234567, { useGrouping: true })).toBe('1,234,567');
  });

  test('handles compact dollar', () => {
    const roundedNumber = 3363690.4;
    const currency = 'USD';
    const compactNumbers = false;
    expect(formatNumber(roundedNumber, { currency, compact: compactNumbers })).toBe(
      '$3,363,690.40'
    );
  });

  test('handles padding, should add zeros to the right', () => {
    // formattedText = formatNumber(roundedNumber, {
    //   minimumFractionDigits: Math.min(minimumFractionDigits, maximumFractionDigits),
    //   maximumFractionDigits: Math.max(minimumFractionDigits, maximumFractionDigits),
    //   useGrouping: numberSeparatorStyle !== null,
    //   compact: compactNumbers
    // });

    expect(formatNumber(1234, { minDecimals: 2 })).toBe('1,234.00');
  });
});
