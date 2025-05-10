import type { FileViewSecondary } from '@/layouts/ChatLayout/ChatLayoutContext/useLayoutConfig';
import { SelectedFile } from '../../interfaces';

export type FileContainerSecondaryProps = {
  selectedFileViewSecondary: FileViewSecondary | undefined;
  selectedFile: SelectedFile | null;
};
