import { useState } from 'react';
import { Button } from '../common/Button';

interface ImageViewerProps {
  totalSlices: number;
  modality: 'CT' | 'PET' | 'Overlay';
}

export function ImageViewer({ totalSlices, modality }: ImageViewerProps) {
  const [slice, setSlice] = useState(1);

  const handleNext = () => setSlice((prev) => Math.min(prev + 1, totalSlices));
  const handlePrev = () => setSlice((prev) => Math.max(prev - 1, 1));

  return (
    <div className="rounded-2xl bg-white p-6 shadow-sm shadow-primary-navy/5">
      <div className="mb-4 flex items-center justify-between">
        <h3 className="text-lg font-semibold text-primary-navy">{modality} Viewer</h3>
        <span className="text-sm font-semibold text-gray-600">
          Slice {slice} of {totalSlices}
        </span>
      </div>
      <div className="flex h-72 items-center justify-center rounded-xl bg-gray-100 text-gray-400">
        <span className="text-sm font-semibold">Medical image viewer placeholder</span>
      </div>
      <div className="mt-4 flex flex-wrap items-center justify-between gap-3">
        <div className="inline-flex items-center gap-2">
          <Button variant="secondary" onClick={handlePrev}>
            Previous Slice
          </Button>
          <Button variant="secondary" onClick={handleNext}>
            Next Slice
          </Button>
        </div>
        <div className="inline-flex items-center gap-2 text-sm text-gray-600">
          <Button variant="tertiary">Zoom In</Button>
          <Button variant="tertiary">Zoom Out</Button>
          <Button variant="tertiary">Reset View</Button>
        </div>
      </div>
    </div>
  );
}
