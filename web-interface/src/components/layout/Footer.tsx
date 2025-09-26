export function Footer() {
  return (
    <footer className="border-t border-gray-100 bg-white">
      <div className="mx-auto flex max-w-7xl flex-col gap-3 px-4 py-6 text-sm text-gray-600 sm:flex-row sm:items-center sm:justify-between">
        <p>&copy; {new Date().getFullYear()} Multi-Modal Cancer Detection Project. All rights reserved.</p>
        <div className="flex items-center gap-4">
          <span className="inline-flex items-center gap-1 text-success-green">
            <span className="h-2 w-2 rounded-full bg-success-green" /> HIPAA-aware system design
          </span>
          <a href="#" className="hover:text-primary-blue" aria-label="Privacy Policy page (link does not navigate)">
            Privacy Policy
          </a>
          <a href="#" className="hover:text-primary-blue" aria-label="Documentation page (link does not navigate)">
            Documentation
          </a>
        </div>
      </div>
    </footer>
  );
}
