import { ReactNode } from 'react';
import clsx from 'clsx';

interface CardProps {
  children: ReactNode;
  className?: string;
}

export function Card({ children, className }: CardProps) {
  return <div className={clsx('rounded-2xl bg-white p-6 shadow-sm shadow-primary-navy/5', className)}>{children}</div>;
}
