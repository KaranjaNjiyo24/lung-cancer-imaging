interface ModalityContribution {
  modality: string;
  percentage: number;
}

interface ModalityBreakdownProps {
  contributions: ModalityContribution[];
}

export function ModalityBreakdown({ contributions }: ModalityBreakdownProps) {
  return (
    <div className="rounded-2xl bg-white p-6 shadow-sm shadow-primary-navy/5">
      <h3 className="text-lg font-semibold text-primary-navy">Modality Contributions</h3>
      <div className="mt-4 space-y-4">
        {contributions.map((item) => (
          <div key={item.modality} className="space-y-2">
            <div className="flex items-center justify-between text-sm font-semibold text-gray-600">
              <span>{item.modality}</span>
              <span>{item.percentage}%</span>
            </div>
            <div className="h-3 w-full overflow-hidden rounded-full bg-gray-100">
              <div
                className="h-full rounded-full bg-primary-blue transition-all"
                style={{ width: `${item.percentage}%` }}
              />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
