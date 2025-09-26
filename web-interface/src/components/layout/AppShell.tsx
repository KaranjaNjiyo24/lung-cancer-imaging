import { ReactNode, Suspense } from 'react';
import { Header } from './Header';
import { Sidebar } from './Sidebar';
import { Footer } from './Footer';

interface AppShellProps {
  children: ReactNode;
}

export function AppShell({ children }: AppShellProps) {
  return (
    <div className="min-h-screen bg-gray-50">
      <Header />
      <div className="mx-auto flex max-w-7xl gap-6 px-4 py-6 sm:px-6 lg:px-8">
        <Sidebar />
        <main className="flex-1">
          <Suspense fallback={<div className="py-16 text-center text-gray-600">Loading...</div>}>
            {children}
          </Suspense>
        </main>
      </div>
      <Footer />
    </div>
  );
}
