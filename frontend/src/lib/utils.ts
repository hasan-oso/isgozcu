import clsx, { type ClassValue } from "clsx";

// Minimal class-name combiner. We don't need tailwind-merge yet.
export function cn(...inputs: ClassValue[]): string {
  return clsx(inputs);
}
