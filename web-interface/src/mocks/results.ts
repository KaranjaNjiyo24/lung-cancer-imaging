export const mockResult = {
  patientId: 'AMC-006',
  analysisDate: 'Today',
  processingTime: '2.3s',
  confidence: 87,
  detection: {
    status: 'Potential malignancy detected',
    confidence: '87% confidence level'
  },
  modalityContributions: [
    { modality: 'CT Imaging', percentage: 65 },
    { modality: 'PET Imaging', percentage: 35 }
  ],
  recommendations: [
    {
      priority: 'high' as const,
      title: 'Immediate Consultation',
      description: 'Schedule oncology consultation within 7 days'
    },
    {
      priority: 'medium' as const,
      title: 'Additional Imaging',
      description: 'Consider MRI for detailed tissue characterization'
    }
  ]
};
