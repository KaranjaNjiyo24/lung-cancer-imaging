import { NavLink } from 'react-router-dom';
import { ClipboardDocumentListIcon, CloudArrowUpIcon, ChartBarIcon, ChartPieIcon } from '@heroicons/react/24/outline';

const links = [
  { to: '/', label: 'Dashboard', icon: ClipboardDocumentListIcon },
  { to: '/upload', label: 'Image Upload', icon: CloudArrowUpIcon },
  { to: '/results', label: 'Results', icon: ChartPieIcon },
  { to: '/analytics', label: 'Analytics', icon: ChartBarIcon }
];

export function Sidebar() {
  return (
    <aside className="hidden w-64 flex-none border-r border-gray-100 bg-white/70 backdrop-blur lg:block">
      <div className="sticky top-16 space-y-1 px-4 py-6">
        {links.map(({ to, label, icon: Icon }) => (
          <NavLink
            key={to}
            to={to}
            className={({ isActive }) =>
              `flex items-center gap-3 rounded-lg px-3 py-2 text-sm font-semibold transition-colors ${
                isActive ? 'bg-primary-blue/10 text-primary-blue' : 'text-gray-600 hover:bg-gray-100'
              }`
            }
          >
            <Icon className="h-5 w-5" />
            {label}
          </NavLink>
        ))}
      </div>
    </aside>
  );
}
