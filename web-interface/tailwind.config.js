/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './index.html',
    './src/**/*.{ts,tsx}'
  ],
  theme: {
    extend: {
      colors: {
        'primary-blue': '#2563eb',
        'primary-navy': '#1e3a8a',
        'accent-teal': '#0d9488',
        'success-green': '#10b981',
        'warning-amber': '#f59e0b',
        'error-red': '#dc2626',
        'gray-50': '#f9fafb',
        'gray-100': '#f3f4f6',
        'gray-600': '#4b5563',
        'gray-900': '#111827'
      },
      fontFamily: {
        sans: ['"Source Sans Pro"', 'Inter', 'ui-sans-serif', 'system-ui']
      }
    }
  },
  plugins: []
};
