import { MetricCard } from '../common/MetricCard';

interface MetricItem {
  label: string;
  value: string;
  trend?: string;
  trendDirection?: 'up' | 'down';
}

interface MetricsGridProps {
  metrics: MetricItem[];
}

export function MetricsGrid({ metrics }: MetricsGridProps) {
  return (
    <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
      {metrics.map((metric) => (
        <MetricCard key={metric.label} {...metric} />
      ))}
    </div>
  );
}
