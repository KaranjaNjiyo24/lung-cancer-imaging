import { ButtonHTMLAttributes, ReactNode } from 'react';
import clsx from 'clsx';

type ButtonVariant = 'primary' | 'secondary' | 'tertiary';

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant;
  startIcon?: ReactNode;
  endIcon?: ReactNode;
}

export function Button({ variant = 'primary', startIcon, endIcon, className, children, ...props }: ButtonProps) {
  const baseClasses = 'inline-flex items-center justify-center gap-2 rounded-full px-5 py-2 text-sm font-semibold transition focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2';

  const variantClasses = {
    primary: 'bg-primary-blue text-white shadow-sm hover:bg-primary-blue/90 focus-visible:outline-primary-blue',
    secondary: 'border border-primary-blue text-primary-blue hover:bg-primary-blue/10 focus-visible:outline-primary-blue',
    tertiary: 'text-primary-blue hover:text-primary-blue/80 focus-visible:outline-primary-blue'
  } as const;

  return (
    <button className={clsx(baseClasses, variantClasses[variant], className)} {...props}>
      {startIcon}
      {children}
      {endIcon}
    </button>
  );
}
