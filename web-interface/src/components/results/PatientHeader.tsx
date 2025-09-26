interface PatientHeaderProps {
  patientId: string;
  analysisDate: string;
  processingTime: string;
  confidence: number;
}

export function PatientHeader({ patientId, analysisDate, processingTime, confidence }: PatientHeaderProps) {
  return (
    <div className="flex flex-col items-start justify-between gap-6 rounded-2xl bg-white p-6 shadow-sm shadow-primary-navy/5 sm:flex-row sm:items-center">
      <div>
        <h2 className="text-2xl font-semibold text-primary-navy">Analysis Results</h2>
        <div className="mt-2 flex flex-wrap gap-3 text-sm text-gray-600">
          <span className="badge">Patient ID: {patientId}</span>
          <span className="badge">Analysis Date: {analysisDate}</span>
          <span className="badge">Processing Time: {processingTime}</span>
        </div>
      </div>
      <div className="flex items-center justify-center rounded-full bg-primary-blue/10 p-6 text-center">
        <div>
          <span className="text-3xl font-semibold text-primary-blue">{confidence}%</span>
          <p className="text-xs font-semibold uppercase tracking-wide text-primary-blue">Confidence</p>
        </div>
      </div>
    </div>
  );
}
