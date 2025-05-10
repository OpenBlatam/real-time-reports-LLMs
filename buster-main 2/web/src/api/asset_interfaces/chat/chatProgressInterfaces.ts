import type {
  BusterChatMessage,
  BusterChatMessageReasoning,
  BusterChatMessageResponse
} from './chatMessageInterfaces';

enum BusterChatStepProgress {
  IN_PROGRESS = 'in_progress',
  COMPLETED = 'completed'
}

type BusterChatStepBase = {
  chat_id: string;
  message_id: string;
  progress: BusterChatStepProgress;
};

export type ChatPost_generatingTitle = {
  title: string;
  title_chunk: string;
} & BusterChatStepBase;

export type ChatPost_generatingMessage = {
  resposnse_message: BusterChatMessageResponse;
} & BusterChatStepBase;

export type ChatPost_generatingReasoning = {
  reasoning: BusterChatMessageReasoning;
} & BusterChatStepBase;

export type ChatPost_complete = {
  message: BusterChatMessage;
} & BusterChatStepBase;
