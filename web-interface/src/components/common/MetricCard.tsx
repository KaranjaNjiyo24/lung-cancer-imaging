import { ReactNode } from 'react';
import { Card } from './Card';
import clsx from 'clsx';

interface MetricCardProps {
  label: string;
  value: string;
  trend?: string;
  trendDirection?: 'up' | 'down';
  icon?: ReactNode;
}

export function MetricCard({ label, value, trend, trendDirection = 'up', icon }: MetricCardProps) {
  return (
    <Card className="flex flex-col gap-2">
      <div className="flex items-center justify-between">
        <p className="text-sm font-semibold text-gray-600 uppercase tracking-wide">{label}</p>
        {icon && <div className="text-primary-blue">{icon}</div>}
      </div>
      <h3 className="text-3xl font-semibold text-primary-navy">{value}</h3>
      {trend && (
        <span
          className={clsx('text-sm font-semibold', {
            'text-success-green': trendDirection === 'up',
            'text-error-red': trendDirection === 'down'
          })}
        >
          {trendDirection === 'up' ? '▲' : '▼'} {trend}
        </span>
      )}
    </Card>
  );
}
