from dataclasses import dataclass
from datetime import date

from lease_lab.config import CONFIG


@dataclass(frozen=True)
class Offer:
    offer_id: str
    rate_bps: int
    contribution_cents: int
    cost_cents: int


def offer(offer_id: str) -> Offer:
    margins = CONFIG["acceptance_contribution_cents"]
    if offer_id not in margins:
        raise ValueError("unknown offer")
    return Offer(
        offer_id, int(offer_id[1:]), margins[offer_id], CONFIG["decision_cost_cents"]
    )


def menu(card_id: str, as_of: str) -> tuple[list[Offer], bool]:
    day = date.fromisoformat(as_of)
    if card_id == "nov-2026":
        ids, start, end = (
            ["r800", "r900", "r1000"],
            date(2026, 11, 1),
            date(2026, 12, 1),
        )
    elif card_id == "dec-2026":
        ids, start, end = ["r850", "r950", "r1050"], date(2026, 12, 1), date(2027, 1, 1)
    else:
        raise ValueError("unknown rate card")
    return [offer(name) for name in ids], start <= day < end
