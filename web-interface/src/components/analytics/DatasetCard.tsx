interface DatasetCardProps {
  name: string;
  description: string;
  stats: string[];
}

export function DatasetCard({ name, description, stats }: DatasetCardProps) {
  return (
    <div className="rounded-2xl bg-white p-6 shadow-sm shadow-primary-navy/5">
      <h4 className="text-lg font-semibold text-primary-navy">{name}</h4>
      <p className="mt-2 text-sm text-gray-600">{description}</p>
      <ul className="mt-4 space-y-2 text-sm text-gray-600">
        {stats.map((item) => (
          <li key={item} className="flex items-center gap-2">
            <span className="h-2 w-2 rounded-full bg-primary-blue" />
            {item}
          </li>
        ))}
      </ul>
    </div>
  );
}
