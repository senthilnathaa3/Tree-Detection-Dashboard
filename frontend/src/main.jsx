import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import './index.css';
import App from './App.jsx';
import LocationValidationPage from './LocationValidationPage.jsx';

function RootRouter() {
  const path = window.location.pathname;
  if (path === '/location-validation') {
    return <LocationValidationPage />;
  }
  return <App />;
}

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <RootRouter />
  </StrictMode>,
);
