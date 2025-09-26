import { ReactNode, useCallback, useState } from 'react';
import clsx from 'clsx';

interface UploadZoneProps {
  label: string;
  description: string;
  accept?: string;
  onFilesSelected: (files: FileList | null) => void;
  icon?: ReactNode;
}

export function UploadZone({ label, description, accept = '.dcm,.dicom', onFilesSelected, icon }: UploadZoneProps) {
  const [isDragging, setDragging] = useState(false);

  const handleDragOver = useCallback((event: React.DragEvent<HTMLLabelElement>) => {
    event.preventDefault();
    setDragging(true);
  }, []);

  const handleDragLeave = useCallback((event: React.DragEvent<HTMLLabelElement>) => {
    event.preventDefault();
    setDragging(false);
  }, []);

  const handleDrop = useCallback(
    (event: React.DragEvent<HTMLLabelElement>) => {
      event.preventDefault();
      setDragging(false);
      onFilesSelected(event.dataTransfer.files);
    },
    [onFilesSelected]
  );

  return (
    <label
      className={clsx(
        'flex flex-col items-center justify-center gap-3 rounded-2xl border-2 border-dashed px-6 py-12 text-center transition',
        isDragging ? 'border-primary-blue bg-primary-blue/5' : 'border-gray-200 bg-white hover:border-primary-blue'
      )}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
    >
      <input
        type="file"
        accept={accept}
        multiple
        className="hidden"
        onChange={(event) => onFilesSelected(event.target.files)}
        aria-labelledby="upload-zone-label"
      />
      <div className="text-primary-blue">{icon}</div>
      <h3 id="upload-zone-label" className="text-lg font-semibold text-primary-navy">{label}</h3>
      <p className="text-sm text-gray-600">{description}</p>
      <p className="text-xs text-gray-500">Supports .dcm and .dicom files</p>
      <span className="btn-secondary mt-4">Browse Files</span>
    </label>
  );
}
