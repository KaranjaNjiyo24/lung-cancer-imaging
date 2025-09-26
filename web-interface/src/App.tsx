import { useRoutes } from 'react-router-dom';
import { AppShell } from './components/layout/AppShell';
import { routes } from './routes';

function App() {
  const element = useRoutes(routes);
  return <AppShell>{element}</AppShell>;
}

export default App;
