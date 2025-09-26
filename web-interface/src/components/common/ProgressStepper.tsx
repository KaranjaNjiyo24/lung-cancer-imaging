interface ProgressStep {
  label: string;
  status: 'pending' | 'in-progress' | 'completed';
}

interface ProgressStepperProps {
  steps: ProgressStep[];
  percentage: number;
}

export function ProgressStepper({ steps, percentage }: ProgressStepperProps) {
  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold text-gray-600 uppercase tracking-wide">Processing Medical Images</h3>
        <span className="text-sm font-semibold text-primary-blue">{percentage}%</span>
      </div>
      <div className="h-2 w-full overflow-hidden rounded-full bg-gray-100">
        <div
          className="h-full rounded-full bg-primary-blue transition-all duration-500"
          style={{ width: `${percentage}%` }}
        />
      </div>
      <div className="grid gap-2 text-sm text-gray-600 sm:grid-cols-4">
        {steps.map((step) => (
          <span
            key={step.label}
            className={`flex items-center gap-2 rounded-full px-3 py-1 ${
              step.status === 'completed'
                ? 'bg-success-green/10 text-success-green'
                : step.status === 'in-progress'
                ? 'bg-primary-blue/10 text-primary-blue'
                : 'bg-gray-100'
            }`}
          >
            <span className="h-2 w-2 rounded-full bg-current" />
            {step.label}
          </span>
        ))}
      </div>
    </div>
  );
}
