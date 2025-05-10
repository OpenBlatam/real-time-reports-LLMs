import { createFormHook } from '@tanstack/react-form';
import {
  NumberField,
  PasswordField,
  SubscribeButton,
  TextField,
  fieldContext,
  formContext
} from './FormBase';

const { useAppForm, withForm } = createFormHook({
  fieldComponents: {
    TextField,
    NumberField,
    PasswordField
  },

  formComponents: {
    SubscribeButton
  },
  fieldContext,
  formContext
});

export { useAppForm, withForm };
