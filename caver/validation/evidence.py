"""CaVER DSE: reference-law calibration and coherent *joint* risk LPs.

A certificate requires frozen whole-episode reference conditional-mean labels,
exchangeable calibration episodes and a separately checked execution prefix.
Calibration against simulated finite-K labels is only simulator-conditional.
Never call this module to infer real-world individual causal responsibility.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import json
import math
import numpy as np
from scipy.optimize import linprog


def risk_kind(key: str) -> str:
    """Group coordinates for simultaneous, constraint-aware calibration."""
    suffix=str(key).rsplit('/',1)[-1]
    if suffix=='ego': return 'ego'
    if suffix=='down': return 'downstream'
    return 'pair'


@dataclass
class Calibration:
    quantile: float
    alpha: float
    n: int
    audit_worlds: int
    mc_delta: float
    mc_buffer: float
    reference_families: list[str]
    predictor_digest: str
    calibration_split: str
    valid: bool = True
    # An offline whole-episode calibration never licenses an adaptively queried
    # release state unless that *same frozen query/hold procedure* was calibrated.
    procedure: str = 'NO_QUERY_NO_HOLD'
    label_scope: str = 'SIMULATED_REFERENCE_CONDITIONAL_MEANS'
    quantiles_by_kind: dict[str,float] = field(default_factory=dict)
    mc_buffers_by_kind: dict[str,float] = field(default_factory=dict)
    # Risk admission uses the one-sided upper residual because an
    # overestimate of a risk coordinate should not widen its upper endpoint.
    # Choice regret still needs the lower endpoint, so both tails are retained.
    upper_quantiles_by_kind: dict[str,float] = field(default_factory=dict)
    lower_quantiles_by_kind: dict[str,float] = field(default_factory=dict)
    nonconformity: str = 'TWO_SIDED_ABSOLUTE_COMPATIBILITY'
    alpha_upper: float | None = None
    alpha_lower: float | None = None

    def interval(self, p, scale=1.0, key=None):
        if not self.valid or not np.isfinite(p) or not (0 <= p <= 1):
            raise ValueError('Invalid reference calibration or risk prediction')
        kind=risk_kind(key) if key is not None else None
        q=self.quantiles_by_kind.get(kind,self.quantile)
        b=self.mc_buffers_by_kind.get(kind,self.mc_buffer)
        q_upper=self.upper_quantiles_by_kind.get(kind,q)
        q_lower=self.lower_quantiles_by_kind.get(kind,q)
        lower_radius = q_lower * scale + b
        upper_radius = q_upper * scale + b
        return (max(0., float(p) - lower_radius),
                min(1., float(p) + upper_radius))

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text()))

    def save(self, path):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.__dict__, indent=2, allow_nan=False))


def conformal_calibrate(samples, alpha=.1, audit_worlds=32, mc_delta=.05,
                        reference_families=None, predictor_digest='NOT_FROZEN',
                        procedure='NO_QUERY_NO_HOLD'):
    """Whole-episode max nonconformity, conservative MC conditional-mean label radius.

    There must be n independent *episodes*, not n agent pairs from one log.
    Hoeffding + union bound controls all calibration-label estimates jointly;
    it does not bound reference-model structural misspecification.
    """
    samples = list(samples)
    if not (0 < alpha < 1 and 0 < mc_delta < 1 and audit_worlds > 0):
        raise ValueError('invalid calibration confidence / Monte Carlo budget')
    if len(samples) < 2:
        raise ValueError('At least two disjoint calibration episodes required')
    keys = [set(row['predicted']) for row in samples]
    if any(not x or x != set(row['audited']) for x, row in zip(keys, samples)):
        raise ValueError('Risk-coordinate mismatch between reference and prediction')
    n = len(samples)
    kinds=sorted({risk_kind(k) for x in keys for k in x})
    # Split the total calibration error between the upper and lower tails
    # before allocating it across risk kinds.  This matches the union bound
    # used by the DSE coverage statement.
    alpha_upper=alpha/2.; alpha_lower=alpha/2.
    alpha_kind=alpha_upper/max(1,len(kinds)); delta_kind=mc_delta/max(1,len(kinds))
    scores_by_kind={kind:[] for kind in kinds}
    upper_scores_by_kind={kind:[] for kind in kinds}
    lower_scores_by_kind={kind:[] for kind in kinds}
    for row in samples:
        row_scores={kind:0. for kind in kinds}
        row_upper={kind:0. for kind in kinds}
        row_lower={kind:0. for kind in kinds}
        for key, p in row['predicted'].items():
            mu_hat = float(row['audited'][key])
            if not np.isfinite(p) or not np.isfinite(mu_hat) or not (0 <= p <= 1 and 0 <= mu_hat <= 1):
                raise ValueError('Risk means must be finite in [0,1]')
            kind=risk_kind(key)
            prediction=float(p)
            row_scores[kind]=max(row_scores[kind],abs(prediction-mu_hat))
            row_upper[kind]=max(row_upper[kind],mu_hat-prediction,0.)
            row_lower[kind]=max(row_lower[kind],prediction-mu_hat,0.)
        for kind in kinds:
            scores_by_kind[kind].append(row_scores[kind])
            upper_scores_by_kind[kind].append(row_upper[kind])
            lower_scores_by_kind[kind].append(row_lower[kind])
    q_by={};b_by={}
    upper_q_by={};lower_q_by={}
    for kind in kinds:
        kk=math.ceil((n+1)*(1-alpha_kind))
        q_by[kind]=1. if kk>n else float(np.sort(scores_by_kind[kind])[kk-1])
        upper_q_by[kind]=1. if kk>n else float(np.sort(upper_scores_by_kind[kind])[kk-1])
        lower_q_by[kind]=1. if kk>n else float(np.sort(lower_scores_by_kind[kind])[kk-1])
        count=sum(1 for x in keys if any(risk_kind(k)==kind for k in x))
        b_by[kind]=math.sqrt(math.log(2*max(1,count)/delta_kind)/(2*audit_worlds))
    q=max(q_by.values(),default=1.);radius=max(b_by.values(),default=0.)
    return Calibration(q, alpha, n, audit_worlds, mc_delta, radius,
                       list(reference_families or []), predictor_digest,
                       'EPISODE_DISJOINT_CALIBRATION', procedure=procedure,
                       quantiles_by_kind=q_by, mc_buffers_by_kind=b_by,
                       upper_quantiles_by_kind=upper_q_by,
                       lower_quantiles_by_kind=lower_q_by,
                       nonconformity='ONE_SIDED_UPPER_AND_LOWER_EPISODE_MAX',
                       alpha_upper=alpha_upper,alpha_lower=alpha_lower)


def risk_intervals(pred, calibration=None):
    if calibration is None:
        return {k: (0., 1.) for k in pred}
    return {k: calibration.interval(p,key=k) for k, p in pred.items()}


@dataclass
class Decision:
    status: str
    chosen: str | None
    candidates: list[str]
    margins: dict
    worst_regret: dict
    reason: str
    certified_under: str
    gamma_robust: dict = field(default_factory=dict)
    gamma_plausible: dict = field(default_factory=dict)
    witnesses: dict = field(default_factory=dict)
    released: bool = False
    risk_only_allowed: bool = False


def _names(actions):
    result = [a.action_id if hasattr(a, 'action_id') else str(a) for a in actions]
    if len(set(result)) != len(result) or not result:
        raise ValueError('Actions must be nonempty with unique IDs')
    return result


class CoherentRiskSet:
    """Compact outer risk polytope: pair <= union <= sum pair.

    Only known mathematical identities are imposed. In particular, independently
    sampled counterfactual pairs are NOT tied by an invented cross-world law.
    Missing competitor predictions use [0,1], never implicit safe or exclusion.
    """
    def __init__(self, predictions, names, pairs, calibration):
        self.names = list(names)
        self.pairs = list(pairs)
        self.keys = [f'{a}/{k}' for a in names for k in ['ego', 'down', *pairs]]
        self.keys += [f'{a}/slack' for a in names if f'{a}/slack' in predictions]
        self.index = {x: i for i, x in enumerate(self.keys)}
        self.bounds = [calibration.interval(float(predictions[key]),key=key) if key in predictions
                       else (0., 1.) for key in self.keys]
        self.A = []
        self.B = []
        for a in names:
            di = self.index[f'{a}/down']
            if not pairs:
                row = self.vec({f'{a}/down': 1})
                self.A.extend([row, -row]); self.B.extend([0., 0.])
            else:
                for p in pairs:
                    self.add({f'{a}/{p}': 1., f'{a}/down': -1.}, 0.)
                self.add({f'{a}/down': 1., **{f'{a}/{p}': -1. for p in pairs}}, 0.)
        self.A = np.asarray(self.A, dtype=float).reshape((-1, len(self.keys)))
        self.B = np.asarray(self.B, dtype=float)
        self._base_check = self.solve(np.zeros(len(self.keys)), maximize=False)
        self.valid = self._base_check['feasible']

    def vec(self, weights):
        out = np.zeros(len(self.keys))
        for k, v in weights.items():
            if k not in self.index:
                raise KeyError(k)
            out[self.index[k]] = float(v)
        return out

    def add(self, weights, bound):
        self.A.append(self.vec(weights))
        self.B.append(float(bound))

    def solve(self, vec, maximize=True, additional=None):
        """LP returns an actual *joint* witness, not coordinatewise mix-and-match."""
        A = self.A
        B = self.B
        if additional:
            A = np.vstack((A, *[r for r, _ in additional]))
            B = np.concatenate((B, [b for _, b in additional]))
        target = -np.asarray(vec) if maximize else np.asarray(vec)
        sol = linprog(target, A_ub=A if len(A) else None,
                      b_ub=B if len(B) else None, bounds=self.bounds, method='highs')
        if sol.status == 2:
            return {'feasible': False, 'reason': 'infeasible'}
        if sol.status != 0:
            raise RuntimeError(f'Risk LP failure {sol.status}: {sol.message}')
        return {'feasible': True, 'value': float(np.dot(vec, sol.x)),
                'risk_vector': {k: float(x) for k, x in zip(self.keys, sol.x)}}


def _support(support, name):
    return bool(support.get(name, False)) if isinstance(support, dict) else support is True


def _prefix(prefix, name):
    return bool(prefix.get(name, False)) if isinstance(prefix, dict) else prefix is True


def _constraints(risk_set, action, reference, pairs, eps_ego, eps_down, eps_inc,
                 eps_slack=None):
    out = [('ego', risk_set.vec({f'{action}/ego': 1.}), eps_ego),
           ('downstream_union', risk_set.vec({f'{action}/down': 1.}), eps_down)]
    for p in pairs:
        out.append((f'increment:{p}', risk_set.vec({} if action == reference else {f'{action}/{p}': 1., f'{reference}/{p}': -1.}), eps_inc))
    if eps_slack is not None and f'{action}/slack' in risk_set.index:
        out.append(('policy_slack_loss', risk_set.vec({f'{action}/slack': 1.}), eps_slack))
    return out


def _objective(risk_set, name, reference, pairs, task_cost, lambda_contact, lambda_inc,
               lambda_slack, pair_weights):
    # Shared reference baseline -sum C_p(reference) is retained: it cancels
    # in action differences but is necessary to report the exact J(a;r).
    weights = {f'{name}/down': lambda_contact}
    for p in pairs:
        w = lambda_inc * pair_weights.get(p, 1.)
        weights[f'{name}/{p}'] = weights.get(f'{name}/{p}', 0.) + w
        refkey = f'{reference}/{p}'
        weights[refkey] = weights.get(refkey, 0.) - w
    if f'{name}/slack' in risk_set.index and all(f'{a}/slack' in risk_set.index for a in risk_set.names):
        weights[f'{name}/slack'] = lambda_slack
    return risk_set.vec(weights), float(task_cost.get(name, 0.))


def choose(pred, actions, pairs, reference='reference', calibration=None,
           eps_ego=.08, eps_down=.12, eps_inc=.10, eps_reg=1.5,
           task_cost=None, support=True, strict_prefix=False,
           allow_risk_only=False, eps_slack=None,
           lambda_contact=5., lambda_inc=0., lambda_slack=0.,
           pair_weights=None, procedure='NO_QUERY_NO_HOLD'):
    """Exact LP DSE: risk gate, Gamma_R and *plausible-safe* Gamma_P.

    Uses all lawful candidates as pointwise feasible comparators regardless of
    robust gate / source support. The selected action needs explicit support.
    The prefix check is an independent caller-supplied assumption, not computed
    from collision probabilities. No certificate without valid calibration.
    """
    names = _names(actions)
    pairs = list(pairs)
    if reference not in names:
        raise ValueError('Reference intervention must be in lawful candidate list')
    if len(set(pairs)) != len(pairs):
        raise ValueError('Repeated non-ego contact pairs')
    if calibration is None or not calibration.valid:
        return Decision('UNCERTIFIED', None, [], {}, {},
                        'no independent terminal reference-law calibration', 'NONE')
    if calibration.procedure != procedure:
        return Decision('UNCERTIFIED', None, [], {}, {},
                        'calibration does not cover the actual query/hold/stopping protocol', 'NONE')
    task_cost = task_cost or {}
    pair_weights = pair_weights or {}
    rs = CoherentRiskSet(pred, names, pairs, calibration)
    if not rs.valid:
        return Decision('UNCERTIFIED', None, [], {}, {}, 'empty coherent risk polytope', 'NONE')
    margins, witnesses, admitted = {}, {}, []
    for a in names:
        rows = _constraints(rs, a, reference, pairs, eps_ego, eps_down, eps_inc, eps_slack)
        failures = []
        upper = {}
        for typ, vec, rhs in rows:
            res = rs.solve(vec)
            upper[typ] = res['value']
            if res['value'] > rhs + 1e-8:
                failures.append(typ)
                witnesses[f'{a}/gate/{typ}'] = {'constraint': typ, 'violation': res['value'] - rhs,
                                                   'risk_world': res['risk_vector']}
        required = [f'{a}/{x}' for x in ['ego', 'down', *pairs]]
        evidence_present = all(k in pred for k in required)
        ok = not failures and _support(support, a) and evidence_present and (eps_slack is None or f'{a}/slack' in rs.index)
        margins[a] = {'gate_pass': ok, 'risk_constraint_upper': upper,
                      'failed_constraints': failures, 'source_supported': _support(support, a) and evidence_present and (eps_slack is None or f'{a}/slack' in rs.index)}
        margins[a].update({'ego_upper': upper['ego'],
                          'downstream_upper': upper['downstream_union'],
                          'max_increment_upper': max((v for k, v in upper.items() if k.startswith('increment:')), default=0.)})
        if ok:
            admitted.append(a)
    if not admitted:
        return Decision('UNCERTIFIED', None, admitted, margins, {},
                        'no robustly risk-admissible action under coherent reference polytope',
                        'REFERENCE_MODEL_ONLY', witnesses=witnesses)
    obj = {a: _objective(rs, a, reference, pairs, task_cost, lambda_contact,
                         lambda_inc, lambda_slack, pair_weights) for a in names}
    gamma_r, gamma_p = {}, {}
    for a in admitted:
        ca, task_a = obj[a]
        best_r = (0., None, None)
        best_p = (0., None, None)
        for b in names:
            cb, task_b = obj[b]
            diff = ca - cb
            delta = task_a - task_b
            if b in admitted:
                sol_r = rs.solve(diff)
                gap_r = max(0., sol_r['value'] + delta)
                if gap_r > best_r[0]:
                    best_r = (gap_r, b, sol_r['risk_vector'])
            constraints = [(v, rhs) for _, v, rhs in _constraints(
                rs, b, reference, pairs, eps_ego, eps_down, eps_inc, eps_slack)]
            sol_p = rs.solve(diff, additional=constraints)
            if sol_p['feasible']:
                gap_p = max(0., sol_p['value'] + delta)
                if gap_p > best_p[0]:
                    best_p = (gap_p, b, sol_p['risk_vector'])
        gamma_r[a] = best_r[0]
        gamma_p[a] = best_p[0]
        if best_p[1] is not None:
            witnesses[f'{a}/plausible_regret'] = {
                'competitor': best_p[1], 'regret': best_p[0], 'risk_world': best_p[2]}
    eligible = [a for a in admitted if _prefix(strict_prefix, a)]
    if not eligible:
        return Decision('UNCERTIFIED', None, admitted, margins, gamma_p,
                        'no independently checked executable prefix', 'NO_EXECUTION_CERTIFICATE',
                        gamma_robust=gamma_r, gamma_plausible=gamma_p, witnesses=witnesses)
    chosen = min(eligible, key=lambda a: (gamma_p[a], obj[a][1], a))
    certified = gamma_p[chosen] <= eps_reg + 1e-8
    if certified:
        return Decision('DECISION-CERTIFIED', chosen, admitted, margins, gamma_p,
                        'robust risk admission and plausible-safe choice within declared reference law',
                        'REFERENCE_MODEL_POPULATION', gamma_r, gamma_p, witnesses, True, allow_risk_only)
    if allow_risk_only:
        return Decision('RISK-ADMISSIBLE', chosen, admitted, margins, gamma_p,
                        'risk-only prefix release; near-optimal action NOT certified',
                        'REFERENCE_MODEL_POPULATION', gamma_r, gamma_p, witnesses, True, True)
    return Decision('RISK-ADMISSIBLE', None, admitted, margins, gamma_p,
                    'risk gate satisfied but choice unresolved; risk-only release disabled',
                    'REFERENCE_MODEL_POPULATION', gamma_r, gamma_p, witnesses, False, False)
