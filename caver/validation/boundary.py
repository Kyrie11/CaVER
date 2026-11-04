"""CBC hybrid-guard exception screening for a *declared* structural reactor.

Computable event-boundary certificate; missing validated bounds or an omitted
reachable guard makes beta=1 (mandatory candidate-conditioned recomputation).
Monte Carlo beta is source-model-conditional, not a real-world safety claim.
"""
from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class AnchorWorld:
    """One paired latent sample z from a fixed anchor response law."""
    guards: dict[str, float]          # relevant ordered, time-causal guards
    clearances: dict[str, float]      # signed swept-footprint margins
    source_family: str


@dataclass(frozen=True)
class VerifiedPerturbation:
    """Externally validated upper bounds conditional on identical prior modes."""
    guard_errors: dict[str, float]
    clearance_errors: dict[str, float]
    complete_guard_graph: bool
    validated_dynamics: bool
    sample_family: str
    provenance: str = 'STRUCTURAL_POLICY_ENVELOPE'


@dataclass(frozen=True)
class BoundaryDecision:
    upper_exception_mass: float
    empirical_exception_mass: float | None
    valid: bool
    reuse: bool
    source_family: str
    reason: str
    first_ambiguous_guard: str | None = None


def _finite_nonnegative(err):
    return err is not None and math.isfinite(float(err)) and float(err) >= 0.


def assess_boundary(worlds, envelope, delta=.05, tolerance=.05):
    worlds = list(worlds)
    family = worlds[0].source_family if worlds else envelope.sample_family
    def unknown(reason):
        return BoundaryDecision(1., None, False, False, family, reason)
    if not worlds:
        return unknown('no i.i.d. matched latent worlds')
    if not (0 < delta < 1 and 0 <= tolerance <= 1):
        raise ValueError('delta and tolerance outside valid range')
    if not envelope.complete_guard_graph or not envelope.validated_dynamics:
        return unknown('incomplete or unvalidated hybrid/dynamics guard envelope')
    if not all(w.source_family == family for w in worlds) or envelope.sample_family != family:
        return unknown('policy family not supported by the declared envelope')
    # Coverage must include every guard on reachable e -> pair paths AND every
    # screened contact pair; a missing entry is not evidence of mode stability.
    guard_ids = set(worlds[0].guards)
    pair_ids = set(worlds[0].clearances)
    if any(set(w.guards)!=guard_ids or set(w.clearances)!=pair_ids for w in worlds):
        return unknown('inconsistent guard/contact support between latent worlds')
    if guard_ids != set(envelope.guard_errors) or pair_ids != set(envelope.clearance_errors):
        return unknown('guard/pair bound missing or extra (guard completeness not auditable)')
    if any(not _finite_nonnegative(x) for x in (*envelope.guard_errors.values(), *envelope.clearance_errors.values())):
        return unknown('an unbounded or invalid incremental bound requires recomputation')
    events=[]
    first=None
    for w in worlds:
        exception = False
        for g, val in w.guards.items():
            if not math.isfinite(float(val)):
                return unknown('invalid anchor guard value')
            if abs(val) <= envelope.guard_errors[g]:
                exception = True
                if first is None:
                    first=g
        for p, margin in w.clearances.items():
            if not math.isfinite(float(margin)):
                return unknown('invalid anchor contact clearance')
            if abs(margin) <= envelope.clearance_errors[p]:
                exception = True
        events.append(int(exception))
    empirical = float(np.mean(events))
    upper = min(1., empirical + math.sqrt(math.log(1/delta)/(2*len(events))))
    return BoundaryDecision(upper, empirical, True, upper <= tolerance, family,
                            'source-law Hoeffding exceptional-mass upper bound',first)


def upper_transfer(anchor_down_contact, bound):
    """C_down(a) <= min(1,C_down(anchor)+beta) only for same audited law."""
    if not 0<=anchor_down_contact<=1:
        raise ValueError('anchor risk outside [0,1]')
    return min(1., anchor_down_contact + bound.upper_exception_mass)


def potential_contrast(v_a, v_b):
    """Shared action potential, exactly antisymmetric and cycle-consistent."""
    return float(v_a)-float(v_b)


def bounded_pair_loss(contact, slack, valid_slack, lam=.25, scale=5.):
    """Paper ell=c+(1-c)lambda_s phi(m_pi), validity-masked."""
    if not 0 <= lam < 1 or not 0 <= contact <= 1:
        raise ValueError('invalid pair loss coefficients')
    if not valid_slack:
        return float(contact)
    if slack is None or not math.isfinite(slack):
        raise ValueError('a valid policy slack must be finite')
    # The deficit penalty is bounded, and is never a contact probability.
    phi = min(1., max(0., -float(slack)/scale))
    return float(contact + (1-contact)*lam*phi)


def frechet_flip_bounds(p_a, p_b):
    """Sharp coupling-free flip-mass bounds from two marginal failure rates."""
    for x in (p_a,p_b):
        if not np.isfinite(x) or not 0 <= x <= 1:
            raise ValueError('Marginal failure probabilities must be in [0,1]')
    return abs(p_a-p_b),min(p_a+p_b,2-p_a-p_b)


def frechet_harm_bounds(p_a, p_b):
    """Sharp bounds on P(F_a=1,F_b=0), not accident responsibility."""
    for x in (p_a,p_b):
        if not np.isfinite(x) or not 0 <= x <= 1:
            raise ValueError('Marginal failure probabilities must be in [0,1]')
    return max(0.,p_a-p_b),min(p_a,1-p_b)
