import { useState } from 'react';
import { PatientHeader } from '../components/results/PatientHeader';
import { ModalityBreakdown } from '../components/results/ModalityBreakdown';
import { ImageViewer } from '../components/results/ImageViewer';
import { RecommendationCard } from '../components/results/RecommendationCard';
import { Card } from '../components/common/Card';
import { mockResult } from '../mocks/results';
import { Button } from '../components/common/Button';

function Results() {
  const [activeViewer, setActiveViewer] = useState<'CT' | 'PET' | 'Overlay'>('CT');

  return (
    <div className="space-y-8">
      <PatientHeader
        patientId={mockResult.patientId}
        analysisDate={mockResult.analysisDate}
        processingTime={mockResult.processingTime}
        confidence={mockResult.confidence}
      />

      <div className="grid gap-6 lg:grid-cols-3">
        <Card className="lg:col-span-2">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-lg font-semibold text-primary-navy">Detection Result</h3>
              <p className="mt-2 text-sm text-gray-600">{mockResult.detection.status}</p>
            </div>
            <span className="rounded-full bg-success-green/10 px-4 py-2 text-sm font-semibold text-success-green">
              {mockResult.detection.confidence}
            </span>
          </div>
          <Button className="mt-6" variant="secondary">
            Export Report
          </Button>
        </Card>
        <ModalityBreakdown contributions={mockResult.modalityContributions} />
      </div>

      <div className="space-y-6">
        <div className="inline-flex rounded-full border border-gray-200 bg-white p-1 shadow-sm">
          {(['CT', 'PET', 'Overlay'] as const).map((modality) => (
            <button
              key={modality}
              onClick={() => setActiveViewer(modality)}
              className={`rounded-full px-4 py-2 text-sm font-semibold transition ${
                activeViewer === modality ? 'bg-primary-blue text-white shadow-sm' : 'text-gray-600 hover:text-primary-blue'
              }`}
            >
              {modality}
            </button>
          ))}
        </div>
        <ImageViewer totalSlices={205} modality={activeViewer} />
      </div>

      <section className="space-y-4">
        <h3 className="text-lg font-semibold text-primary-navy">AI-Generated Treatment Suggestions</h3>
        <div className="grid gap-4 md:grid-cols-2">
          {mockResult.recommendations.map((recommendation) => (
            <RecommendationCard key={recommendation.title} {...recommendation} />
          ))}
        </div>
        <div className="flex items-center gap-2 rounded-2xl border border-warning-amber/40 bg-warning-amber/10 px-4 py-3 text-sm text-warning-amber">
          <span className="h-2 w-2 rounded-full bg-warning-amber" />
          These are AI-generated suggestions. Always consult with qualified medical professionals for clinical decisions.
        </div>
      </section>
    </div>
  );
}

export default Results;
