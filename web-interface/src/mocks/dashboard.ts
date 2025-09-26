export const dashboardMetrics = [
  { label: 'Detection Accuracy', value: '92.5%' },
  { label: 'Patient Cases Analyzed', value: '48' },
  { label: 'Imaging Modalities', value: '2' }
];

export const quickActions = [
  {
    title: 'Upload Medical Images',
    description: 'Analyze CT and PET scans for cancer detection',
    to: '/upload',
    variant: 'primary' as const
  },
  {
    title: 'View Performance',
    description: 'Model metrics and validation results',
    to: '/analytics',
    variant: 'secondary' as const
  },
  {
    title: 'Research Details',
    description: 'Methodology and technical implementation',
    to: '/about',
    variant: 'tertiary' as const
  }
];
