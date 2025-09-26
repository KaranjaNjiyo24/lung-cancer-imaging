import { lazy } from 'react';
import { RouteObject } from 'react-router-dom';

const DashboardPage = lazy(() => import('./pages/Dashboard'));
const UploadPage = lazy(() => import('./pages/Upload'));
const ResultsPage = lazy(() => import('./pages/Results'));
const AnalyticsPage = lazy(() => import('./pages/Analytics'));
const AboutPage = lazy(() => import('./pages/About'));

export const routes: RouteObject[] = [
  { path: '/', element: <DashboardPage /> },
  { path: '/upload', element: <UploadPage /> },
  { path: '/results', element: <ResultsPage /> },
  { path: '/analytics', element: <AnalyticsPage /> },
  { path: '/about', element: <AboutPage /> }
];
