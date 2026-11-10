"""Annuity with balloon in addition to the last monthly instalment.

Exact rational annuity arithmetic, nominal annual rate / 12. Round displayed
monthly payment half up. The last instalment settles the rounding difference,
in addition to the balloon. Model utilities use binary64 cents separately.
No lender claim.
"""

from fractions import Fraction
from typing import NamedTuple

from lease_lab.contracts import Finance


class Schedule(NamedTuple):
    monthly_cents: int
    last_instalment_cents: int
    total_cents: int
    interest_cents: int


def half_up(value: Fraction) -> int:
    return (2 * value.numerator + value.denominator) // (2 * value.denominator)


def payment_schedule(finance: Finance, rate_bps: int) -> Schedule:
    if type(rate_bps) is not int or not 0 <= rate_bps <= 10000:
        raise ValueError("nominal rate must be integer basis points in [0,10000]")
    principal, balloon, months = (
        finance.principal_cents,
        finance.balloon_cents,
        finance.term_months,
    )
    rate = Fraction(rate_bps, 120000)
    if rate == 0:
        instalment = Fraction(principal - balloon, months)
    else:
        growth = (1 + rate) ** months
        instalment = (principal * growth - balloon) * rate / (growth - 1)
    monthly = half_up(instalment)
    if not rate:
        last = Fraction(principal - balloon - monthly * (months - 1))
    else:
        growth_before = (1 + rate) ** (months - 1)
        balance = principal * growth_before - monthly * (growth_before - 1) / rate
        last = balance * (1 + rate) - balloon
    final = half_up(last)
    if final < 0:
        raise ValueError(
            "rounded instalments overpay this amount/term; choose representable financing terms"
        )
    total = monthly * (months - 1) + final + balloon
    return Schedule(monthly, final, total, total - principal)


def payment(finance: Finance, rate_bps: int) -> tuple[int, int, int]:
    schedule = payment_schedule(finance, rate_bps)
    return schedule.monthly_cents, schedule.total_cents, schedule.interest_cents
