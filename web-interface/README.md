# Web Interface Frontend

This directory contains the React-based frontend for the Multi-Modal Cancer Detection system.

## Component Architecture

```
src/
├── App.tsx                # Root application shell with routing
├── main.tsx               # Entry point
├── assets/                # Static assets and icons
├── components/
│   ├── layout/
│   │   ├── Header.tsx     # Top navigation with branding and nav links
│   │   ├── Sidebar.tsx    # Secondary navigation for desktop
│   │   └── Footer.tsx     # Global footer with compliance info
│   ├── common/
│   │   ├── Button.tsx     # Reusable button variants (primary/secondary/tertiary)
│   │   ├── Card.tsx       # Base card container with variants
│   │   ├── MetricCard.tsx # KPI display component
│   │   ├── ProgressStepper.tsx # Upload progress visualization
│   │   └── Tabs.tsx       # Accessible tab component for modality switching
│   ├── upload/
│   │   └── UploadZone.tsx # Drag-and-drop upload zones for CT/PET
│   ├── results/
│   │   ├── PatientHeader.tsx   # Patient metadata + confidence score
│   │   ├── ModalityBreakdown.tsx # Contribution bars per modality
│   │   ├── ImageViewer.tsx     # Placeholder viewer shell with controls
│   │   └── RecommendationCard.tsx
│   └── analytics/
│       ├── MetricsGrid.tsx     # Overview metrics
│       ├── ChartCard.tsx       # Wrapper for charts
│       └── DatasetCard.tsx     # Dataset information block
├── pages/
│   ├── Dashboard.tsx
│   ├── Upload.tsx
│   ├── Results.tsx
│   └── Analytics.tsx
├── routes.tsx             # React Router configuration
├── theme.ts               # Tailwind extension + design tokens
└── styles/
    └── index.css          # Tailwind directives and base styles
```

The UI is implemented with React 18, TypeScript, Tailwind CSS, and Chart.js. Pages are routed via React Router, and state management is handled locally for the prototype with opportunities to integrate Redux or Zustand for production.

## Getting Started

1. `npm install` – install dependencies
2. `npm run dev` – start the Vite dev server (default http://localhost:5173)
3. `npm run build` – generate a production build
4. `npm run preview` – preview the production bundle locally

## Design Tokens

Color palette, typography scale, and spacing rules follow the design guide included with the project. Tailwind configuration exposes these tokens for use in class utilities.

## Mock Data

The prototype ships with mock data objects (see `src/mocks/`) to enable offline demos without backend integration. Replace these stubs with API hooks once the FastAPI endpoints are available.

## Next Steps

1. Connect upload components to the backend `/api/upload/dicom` endpoint and expose inference progress via websockets or long polling.
2. Replace mock analytics data with live metrics sourced from the ML pipeline.
3. Integrate a medical imaging viewer such as Cornerstone.js or OHIF for slice navigation and overlay blending.
4. Harden accessibility with automated testing (axe-core) and end-to-end validation.
