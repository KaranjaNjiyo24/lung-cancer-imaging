import { Link, NavLink } from 'react-router-dom';
import { Bars3BottomRightIcon } from '@heroicons/react/24/outline';
import { useState } from 'react';

const navLinks = [
  { to: '/', label: 'Dashboard' },
  { to: '/upload', label: 'Image Upload' },
  { to: '/results', label: 'Results' },
  { to: '/analytics', label: 'Analytics' }
];

export function Header() {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <header className="sticky top-0 z-40 bg-white/90 backdrop-blur border-b border-gray-100">
      <div className="mx-auto flex h-16 max-w-7xl items-center justify-between px-4 sm:px-6 lg:px-8">
        <Link to="/" className="flex items-center gap-2 text-primary-navy font-semibold text-lg">
          <span className="inline-flex h-8 w-8 items-center justify-center rounded-lg bg-primary-blue text-white font-bold">AI</span>
          <span>Multi-Modal Cancer Detection</span>
        </Link>
        <nav className="hidden md:flex items-center gap-6">
          {navLinks.map((link) => (
            <NavLink
              key={link.to}
              to={link.to}
              className={({ isActive }) =>
                `text-sm font-semibold transition-colors ${
                  isActive ? 'text-primary-blue' : 'text-gray-600 hover:text-primary-blue'
                }`
              }
            >
              {link.label}
            </NavLink>
          ))}
        </nav>
        <button
          className="md:hidden inline-flex items-center justify-center rounded-md border border-gray-200 p-2 text-gray-600"
          onClick={() => setIsOpen((prev) => !prev)}
        >
          <Bars3BottomRightIcon className="h-6 w-6" />
        </button>
      </div>
      {isOpen && (
        <nav className="md:hidden border-t border-gray-100 bg-white">
          <div className="space-y-1 px-4 py-3">
            {navLinks.map((link) => (
              <NavLink
                key={link.to}
                to={link.to}
                onClick={() => setIsOpen(false)}
                className={({ isActive }) =>
                  `block rounded-md px-3 py-2 text-sm font-semibold ${
                    isActive ? 'bg-primary-blue/10 text-primary-blue' : 'text-gray-600 hover:bg-gray-100'
                  }`
                }
              >
                {link.label}
              </NavLink>
            ))}
          </div>
        </nav>
      )}
    </header>
  );
}
