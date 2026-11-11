// Must load first: LangGraph needs these web APIs before its modules are evaluated.
import './src/polyfills';

import { registerRootComponent } from 'expo';
import App from './App';

registerRootComponent(App);
