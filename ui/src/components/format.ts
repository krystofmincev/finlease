export const euro = (cents: number, decimals = 0) => new Intl.NumberFormat('en-IE', { style: 'currency', currency: 'EUR', maximumFractionDigits: decimals, minimumFractionDigits: decimals }).format(cents / 100);
export const percent = (value: number, decimals = 0) => `${(value * 100).toFixed(decimals)}%`;
export const segmentName = { new: 'New business', growing: 'Growing', established: 'Established', restart: 'Restart · rare' };

export function reasonText(reason: string) {
  const reasons: Record<string, string> = {
    highest_posterior_mean_contribution: 'Highest estimated contribution, before accounting for uncertainty about improvement.',
    insufficient_lower_bound_improvement: 'The lower bound of improvement does not clear the fictional €20 margin. Keep the supported incumbent.',
    supported_lower_bound_exceeds_margin: 'The lower bound of improvement clears the fictional €20 margin, with observed support for this offer.',
    numerically_borderline_improvement: 'The lower bound is too close to the switching margin to resolve reliably with these draws. Review the case.',
    insufficient_observed_support: 'There is insufficient observed support for automated advice. Manual review costs €15; no future conversion is credited.',
    incumbent_unavailable_or_unsupported: 'The incumbent is unavailable or unsupported. This case needs manual review.',
  };
  return reasons[reason] ?? reason.replaceAll('_', ' ');
}
