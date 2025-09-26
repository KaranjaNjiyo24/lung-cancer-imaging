import { Link } from 'react-router-dom';
import { ArrowRightIcon, CloudArrowUpIcon, ChartBarIcon, BookOpenIcon } from '@heroicons/react/24/outline';
import { Card } from '../components/common/Card';
import { dashboardMetrics, quickActions } from '../mocks/dashboard';

const actionIcons = {
  primary: <CloudArrowUpIcon className="h-6 w-6" />,
  secondary: <ChartBarIcon className="h-6 w-6" />,
  tertiary: <BookOpenIcon className="h-6 w-6" />
};

function Dashboard() {
  return (
    <div className="space-y-8">
      <section className="hero-gradient overflow-hidden rounded-3xl border border-primary-blue/10 bg-white/80 px-6 py-10 shadow-lg shadow-primary-navy/5">
        <div className="mb-6 inline-flex items-center gap-2 rounded-full bg-primary-blue/10 px-4 py-1 text-xs font-semibold uppercase tracking-wide text-primary-blue">
          <span>Masters Research Project</span>
          <span className="text-primary-navy">Multi-Modal Cancer Detection</span>
        </div>
        <h1 className="text-3xl font-semibold text-primary-navy sm:text-4xl">AI-Powered Cancer Detection System</h1>
        <p className="mt-4 max-w-2xl text-lg text-gray-600">
          Advanced multi-modal fusion combining CT, PET imaging with genomic data to support medical supervisors and oncology teams.
        </p>
        <div className="mt-8 grid gap-4 sm:grid-cols-3">
          {dashboardMetrics.map((metric) => (
            <Card key={metric.label} className="metric-card">
              <h3>{metric.value}</h3>
              <p className="text-sm font-semibold text-gray-600">{metric.label}</p>
            </Card>
          ))}
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-semibold text-primary-navy">Quick Actions</h2>
        <div className="mt-4 grid gap-6 md:grid-cols-3">
          {quickActions.map((action) => (
            <Card key={action.title} className={`border-l-4 ${
              action.variant === 'primary'
                ? 'border-accent-teal'
                : action.variant === 'secondary'
                ? 'border-primary-blue'
                : 'border-gray-200'
            }`}>
              <div className="flex flex-col gap-4">
                <div className="inline-flex h-12 w-12 items-center justify-center rounded-full bg-primary-blue/10 text-primary-blue">
                  {actionIcons[action.variant]}
                </div>
                <div>
                  <h3 className="text-lg font-semibold text-primary-navy">{action.title}</h3>
                  <p className="mt-2 text-sm text-gray-600">{action.description}</p>
                </div>
                <Link
                  to={action.to}
                  className="inline-flex items-center gap-2 text-sm font-semibold text-primary-blue hover:text-primary-blue/80"
                >
                  {action.variant === 'tertiary' ? 'Learn More' : 'Start Now'}
                  <ArrowRightIcon className="h-4 w-4" />
                </Link>
              </div>
            </Card>
          ))}
        </div>
      </section>
    </div>
  );
}

export default Dashboard;
