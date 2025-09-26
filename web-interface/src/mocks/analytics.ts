export const analyticsMetrics = [
  { label: 'Overall Accuracy', value: '92.5%', trend: '+2.3% vs baseline', trendDirection: 'up' as const },
  { label: 'Precision', value: '89.2%', trend: '+1.8% vs baseline', trendDirection: 'up' as const },
  { label: 'Recall', value: '94.1%', trend: '+3.2% vs baseline', trendDirection: 'up' as const },
  { label: 'F1-Score', value: '91.6%', trend: '+2.5% vs baseline', trendDirection: 'up' as const }
];

export const datasetStats = {
  name: 'NSCLC-Radiogenomics',
  description: '48 multi-modal patient cases with paired CT and PET scans.',
  stats: ['CT Scans: 48 series', 'PET Scans: 48 series', 'Total Images: ~10,000 slices']
};
