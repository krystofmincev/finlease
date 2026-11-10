export function Interval({ low, high, mean, min = 0, max = 1, label }: { low: number; high: number; mean: number; min?: number; max?: number; label: string }) {
  const position = (n: number) => `${Math.max(0, Math.min(100, (n - min) / (max - min) * 100))}%`;
  return <div className="interval" role="img" aria-label={label}>
    <span className="interval-band" style={{ left: position(low), width: `${Math.max(0, Math.min(100, (high - low) / (max - min) * 100))}%` }} />
    <span className="interval-point" style={{ left: position(mean) }} />
  </div>;
}
