import { useState } from 'react';
import { UploadZone } from '../components/upload/UploadZone';
import { Tabs } from '../components/common/Tabs';
import { ProgressStepper } from '../components/common/ProgressStepper';
import { CloudArrowUpIcon, PhotoIcon } from '@heroicons/react/24/outline';

const modalityOptions = [
  { id: 'multi', label: 'Multi-Modal (CT + PET)' },
  { id: 'ct', label: 'CT Only' },
  { id: 'pet', label: 'PET Only' }
];

function Upload() {
  const [mode, setMode] = useState<'multi' | 'ct' | 'pet'>('multi');
  const [progress, setProgress] = useState(0);

  const handleFiles = (files: FileList | null) => {
    if (!files) return;
    setProgress(10);
    setTimeout(() => setProgress(45), 300);
    setTimeout(() => setProgress(75), 600);
    setTimeout(() => setProgress(100), 900);
  };

  const steps = [
    { label: 'Uploading Files', status: progress >= 10 ? 'completed' : 'pending' },
    { label: 'DICOM Validation', status: progress >= 45 ? 'completed' : progress >= 10 ? 'in-progress' : 'pending' },
    { label: 'Image Preprocessing', status: progress >= 75 ? 'completed' : progress >= 45 ? 'in-progress' : 'pending' },
    { label: 'Model Inference', status: progress >= 100 ? 'completed' : progress >= 75 ? 'in-progress' : 'pending' }
  ] as const;

  return (
    <div className="space-y-8">
      <section className="flex flex-col gap-6 rounded-3xl bg-white p-6 shadow-sm shadow-primary-navy/5">
        <div className="flex flex-col justify-between gap-4 sm:flex-row sm:items-center">
          <div>
            <h2 className="text-2xl font-semibold text-primary-navy">Medical Image Analysis</h2>
            <p className="text-sm text-gray-600">Upload CT and PET DICOM series for multi-modal cancer detection.</p>
          </div>
          <Tabs options={modalityOptions} value={mode} onChange={(val) => setMode(val as typeof mode)} />
        </div>
        <div className="grid gap-6 md:grid-cols-2">
          {(mode === 'multi' || mode === 'ct') && (
            <UploadZone
              label="Upload CT Scan"
              description="Drag DICOM files here or click to browse"
              onFilesSelected={handleFiles}
              icon={<CloudArrowUpIcon className="h-10 w-10" />}
            />
          )}
          {(mode === 'multi' || mode === 'pet') && (
            <UploadZone
              label="Upload PET Scan"
              description="Drag DICOM files here or click to browse"
              onFilesSelected={handleFiles}
              icon={<PhotoIcon className="h-10 w-10" />}
            />
          )}
        </div>
      </section>

      {progress > 0 && (
        <section className="rounded-3xl bg-white p-6 shadow-sm shadow-primary-navy/5">
          <ProgressStepper steps={steps} percentage={progress} />
        </section>
      )}
    </div>
  );
}

export default Upload;
