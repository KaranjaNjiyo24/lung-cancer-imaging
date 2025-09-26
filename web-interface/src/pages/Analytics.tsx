import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Tooltip,
  Filler,
  ArcElement,
  Legend,
  ChartOptions
} from 'chart.js';
import { Line, Doughnut } from 'react-chartjs-2';
import { MetricsGrid } from '../components/analytics/MetricsGrid';
import { ChartCard } from '../components/analytics/ChartCard';
import { DatasetCard } from '../components/analytics/DatasetCard';
import { analyticsMetrics, datasetStats } from '../mocks/analytics';

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Tooltip, Filler, ArcElement, Legend);

const trainingData = {
  labels: ['Epoch 1', 'Epoch 2', 'Epoch 3', 'Epoch 4', 'Epoch 5', 'Epoch 6'],
  datasets: [
    {
      label: 'Accuracy',
      data: [0.82, 0.85, 0.88, 0.9, 0.91, 0.925],
      borderColor: '#2563eb',
      backgroundColor: 'rgba(37, 99, 235, 0.2)',
      fill: true,
      tension: 0.4
    },
    {
      label: 'Loss',
      data: [0.45, 0.39, 0.33, 0.28, 0.24, 0.2],
      borderColor: '#0d9488',
      backgroundColor: 'rgba(13, 148, 136, 0.2)',
      fill: true,
      tension: 0.4,
      yAxisID: 'y1'
    }
  ]
};

const trainingOptions: ChartOptions<'line'> = {
  responsive: true,
  interaction: { mode: 'index' as const, intersect: false },
  stacked: false,
  plugins: { legend: { display: true, position: 'bottom' as const } },
  scales: {
    y: { beginAtZero: true, ticks: { callback: (value) => `${Math.round(Number(value) * 100)}%` } },
    y1: {
      position: 'right' as const,
      grid: { drawOnChartArea: false },
      ticks: {
        callback: (value) => `${Math.round(Number(value) * 100)}%`
      }
    }
  }
};

const rocData = {
  labels: ['0', '0.2', '0.4', '0.6', '0.8', '1.0'],
  datasets: [
    {
      label: 'ROC Curve',
      data: [0, 0.45, 0.68, 0.82, 0.93, 1],
      borderColor: '#2563eb',
      fill: true,
      backgroundColor: 'rgba(37, 99, 235, 0.2)',
      tension: 0.4
    }
  ]
};

const rocOptions: ChartOptions<'line'> = {
  responsive: true,
  plugins: { legend: { display: false } },
  scales: {
    x: { title: { display: true, text: 'False Positive Rate' } },
    y: { title: { display: true, text: 'True Positive Rate' } }
  }
};

const confusionData = {
  labels: ['True Positive', 'False Positive', 'True Negative', 'False Negative'],
  datasets: [
    {
      label: 'Counts',
      data: [42, 3, 38, 4],
      backgroundColor: ['#10b981', '#f59e0b', '#2563eb', '#dc2626']
    }
  ]
};

const confusionOptions: ChartOptions<'doughnut'> = {
  responsive: true,
  plugins: { legend: { display: false } }
};

function Analytics() {
  return (
    <div className="space-y-8">
      <section className="space-y-4">
        <h2 className="text-2xl font-semibold text-primary-navy">Model Performance Metrics</h2>
        <MetricsGrid metrics={analyticsMetrics} />
      </section>

      <section className="grid gap-6 xl:grid-cols-3">
        <ChartCard title="Training Progress">
          <Line data={trainingData} options={trainingOptions} />
        </ChartCard>
        <ChartCard title="Confusion Matrix">
          <Doughnut data={confusionData} options={confusionOptions} />
        </ChartCard>
        <ChartCard title="ROC Curve">
          <Line data={rocData} options={rocOptions} />
        </ChartCard>
      </section>

      <section className="space-y-4">
        <h3 className="text-lg font-semibold text-primary-navy">Dataset Statistics</h3>
        <DatasetCard {...datasetStats} />
      </section>
    </div>
  );
}

export default Analytics;
