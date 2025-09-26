interface RecommendationCardProps {
  priority: 'high' | 'medium' | 'low';
  title: string;
  description: string;
}

export function RecommendationCard({ priority, title, description }: RecommendationCardProps) {
  const priorityStyles: Record<RecommendationCardProps['priority'], string> = {
    high: 'bg-error-red/10 text-error-red',
    medium: 'bg-warning-amber/10 text-warning-amber',
    low: 'bg-success-green/10 text-success-green'
  };

  return (
    <div className="rounded-2xl bg-white p-6 shadow-sm shadow-primary-navy/5">
      <span className={`inline-flex items-center rounded-full px-3 py-1 text-xs font-semibold uppercase tracking-wide ${priorityStyles[priority]}`}>
        {priority} priority
      </span>
      <h4 className="mt-3 text-lg font-semibold text-primary-navy">{title}</h4>
      <p className="mt-2 text-sm text-gray-600">{description}</p>
    </div>
  );
}
