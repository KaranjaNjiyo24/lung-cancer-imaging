import { ReactNode } from 'react';
import { Card } from '../common/Card';

interface ChartCardProps {
  title: string;
  children: ReactNode;
}

export function ChartCard({ title, children }: ChartCardProps) {
  return (
    <Card>
      <h3 className="text-lg font-semibold text-primary-navy">{title}</h3>
      <div className="mt-4 h-64">{children}</div>
    </Card>
  );
}
