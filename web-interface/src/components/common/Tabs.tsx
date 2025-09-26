import { ReactNode } from 'react';
import clsx from 'clsx';

export interface TabOption<T extends string = string> {
  id: T;
  label: string;
}

interface TabsProps<T extends string = string> {
  options: TabOption<T>[];
  value: T;
  onChange: (value: T) => void;
}

export function Tabs<T extends string>({ options, value, onChange }: TabsProps<T>) {
  return (
    <div className="inline-flex rounded-full border border-gray-200 bg-white p-1 shadow-sm">
      {options.map((option) => (
        <button
          key={option.id}
          type="button"
          onClick={() => onChange(option.id)}
          className={clsx(
            'rounded-full px-4 py-2 text-sm font-semibold transition',
            value === option.id ? 'bg-primary-blue text-white shadow-sm' : 'text-gray-600 hover:text-primary-blue'
          )}
        >
          {option.label}
        </button>
      ))}
    </div>
  );
}
