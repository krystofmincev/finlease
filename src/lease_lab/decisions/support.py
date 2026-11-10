from lease_lab.artifacts.store import SupportProfile
from lease_lab.config import CONFIG
from lease_lab.contracts import Context, Finance, Support
from lease_lab.models.beta_cells import BetaState, cell_for


def support_for(
    profile: SupportProfile,
    state: BetaState,
    context: Context,
    finance: Finance,
    offer_id: str,
    card_id: str,
    eligible: bool,
) -> Support:
    cell = cell_for(state, context.segment, offer_id)
    responses = cell.accepted + cell.refused
    bounds = profile.by_segment.get(context.segment, profile)
    within = all(
        getattr(bounds, name).minimum
        <= getattr(context, name)
        <= getattr(bounds, name).maximum
        for name in (
            "business_age_years",
            "monthly_turnover_cents",
            "existing_debt_burden",
        )
    )
    finance_supported = all(
        getattr(finance, name) == CONFIG[name]
        for name in ("principal_cents", "term_months", "balloon_cents")
    )
    reasons = []
    if not eligible:
        reasons.append("expired_rate_card")
    if card_id not in profile.observed_card_ids:
        reasons.append("unobserved_card_version")
    if not within:
        reasons.append("outside_training_range")
    if not finance_supported:
        reasons.append("unobserved_finance_terms")
    if responses == 0:
        reasons.append("no_observed_support")
    elif responses < CONFIG["minimum_responses"]:
        reasons.append("low_response_count")
    blocked = (
        not eligible
        or not within
        or not finance_supported
        or card_id not in profile.observed_card_ids
    )
    status = (
        "blocked"
        if blocked
        else "prior_only"
        if responses == 0
        else "sparse"
        if responses < CONFIG["minimum_responses"]
        else "supported"
    )
    return Support(
        state=status,
        automated=not reasons,
        exposure_count=cell.exposures,
        response_count=responses,
        accepted_count=cell.accepted,
        pending_count=cell.pending,
        observed_card_ids=profile.observed_card_ids,
        observed_policy_versions=profile.observed_policy_versions,
        observed_rate_bps=profile.observed_rate_bps,
        within_training_range=within,
        reasons=reasons,
    )
