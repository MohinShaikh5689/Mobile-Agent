import { useColorScheme } from 'react-native';

export interface Theme {
  bg: string;
  surface: string;
  text: string;
  muted: string;
  border: string;
  accent: string;
  accentText: string;
  userBubble: string;
  error: string;
}

const light: Theme = {
  bg: '#f6f6f4',
  surface: '#ffffff',
  text: '#1c1c1a',
  muted: '#6b6b66',
  border: '#e2e2dd',
  accent: '#2f5d50',
  accentText: '#ffffff',
  userBubble: '#e3ece8',
  error: '#b3261e',
};

const dark: Theme = {
  bg: '#141413',
  surface: '#1f1f1d',
  text: '#ecece8',
  muted: '#9b9b95',
  border: '#33332f',
  accent: '#7fb7a4',
  accentText: '#0f1f1a',
  userBubble: '#25332e',
  error: '#f2b8b5',
};

export function useTheme(): Theme {
  return useColorScheme() === 'dark' ? dark : light;
}
