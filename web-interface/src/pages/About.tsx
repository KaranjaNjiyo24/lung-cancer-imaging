import { Card } from '../components/common/Card';

function About() {
  return (
    <div className="space-y-8">
      <Card>
        <h2 className="text-2xl font-semibold text-primary-navy">Research Overview</h2>
        <p className="mt-4 text-sm text-gray-600">
          This Masters research project explores multi-modal deep learning architectures that combine CT and PET imaging with
          radiogenomic metadata to support oncology decision-making. The interface presented here demonstrates the user
          experience for medical supervisors reviewing AI-assisted detection results.
        </p>
      </Card>
      <div className="grid gap-6 md:grid-cols-2">
        <Card>
          <h3 className="text-lg font-semibold text-primary-navy">Model Architecture</h3>
          <ul className="mt-3 space-y-2 text-sm text-gray-600">
            <li>• Multi-modal 3D CNN encoders for CT and PET sequences</li>
            <li>• Attention-based fusion layer to balance modality contributions</li>
            <li>• Classification head supporting binary or multi-class tasks</li>
          </ul>
        </Card>
        <Card>
          <h3 className="text-lg font-semibold text-primary-navy">Compliance & Privacy</h3>
          <p className="mt-3 text-sm text-gray-600">
            The system emphasises patient data protection with secure DICOM handling, audit logging, and export controls
            designed to align with HIPAA and GDPR expectations for clinical deployments.
          </p>
        </Card>
      </div>
    </div>
  );
}

export default About;
