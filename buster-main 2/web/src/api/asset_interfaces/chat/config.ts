export type FileType = 'metric' | 'dashboard' | 'reasoning'; //'dataset' | 'collection' | | 'term' | 'value'

export type ThoughtFileType =
  | 'metric'
  | 'dashboard'
  | 'collection'
  | 'dataset'
  | 'term'
  | 'topic'
  | 'value'
  | 'empty';

export type AllFileTypes = ThoughtFileType | FileType;
