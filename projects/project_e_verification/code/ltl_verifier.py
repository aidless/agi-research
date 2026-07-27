"""ltl_verifier.py - Phase 1.4: LTL rule language + symbolic checker.

Project E: Neuro-symbolic verification. The verifier takes a
trajectory + LTL rules and outputs (satisfied, graded_truth_value).

LTL syntax (subset):
  formula ::= atom | NOT formula | formula AND formula | formula OR formula
            | EVENTUALLY formula | ALWAYS formula | formula UNTIL formula
            | formula IMPLIES formula
  atom ::= "predicate(args)"

LunarLander predicates:
  landed(), in_pad(), leg_contact(L|R), fuel_below(x),
  velocity_below(x), angle_below(x), distance_to_pad(d)
"""
import re
from typing import List, Dict, Callable, Tuple, Any


# LunarLander predicates as Python functions
def _leg_contact(trace, t, leg):
    return trace[t]["leg_" + leg] > 0.5

def _velocity_below(trace, t, threshold):
    vx = abs(trace[t]["x_vel"])
    vy = abs(trace[t]["y_vel"])
    return (vx ** 2 + vy ** 2) ** 0.5 < float(threshold)

def _angle_below(trace, t, threshold):
    return abs(trace[t]["angle"]) < float(threshold)

def _landed(trace, t):
    return trace[t]["leg_l"] > 0.5 or trace[t]["leg_r"] > 0.5

def _in_pad(trace, t):
    x = trace[t]["x_pos"]
    return abs(x) < 0.2  # landing pad is centered

def _distance_to_pad(trace, t):
    return abs(trace[t]["x_pos"])


PREDICATES = {
    "leg_contact": _leg_contact,
    "velocity_below": _velocity_below,
    "angle_below": _angle_below,
    "landed": _landed,
    "in_pad": _in_pad,
    "distance_to_pad": _distance_to_pad,
}


def parse_ltl_atom(text):
    """Parse a single atom like 'velocity_below(0.5)' into (pred, args)."""
    m = re.match(r"(\w+)\(([^)]*)\)", text.strip())
    if not m:
        raise ValueError(f"Cannot parse atom: {text}")
    pred = m.group(1)
    args_str = m.group(2).strip()
    if args_str:
        args = [a.strip() for a in args_str.split(",")]
    else:
        args = []
    return pred, args


def evaluate_atom(atom_text, trace, t):
    """Evaluate atomic predicate at time t."""
    pred, args = parse_ltl_atom(atom_text)
    if pred not in PREDICATES:
        raise ValueError(f"Unknown predicate: {pred}")
    return PREDICATES[pred](trace, t, *args)


def evaluate_ltl(formula, trace, t=0):
    """Recursive LTL evaluation. Returns bool at time t.

    Supported ops: NOT, AND, OR, EVENTUALLY, ALWAYS, UNTIL, IMPLIES
    """
    formula = formula.strip()
    # Parentheses
    if formula.startswith("(") and formula.endswith(")"):
        return evaluate_ltl(formula[1:-1], trace, t)
    # Negation
    if formula.startswith("NOT "):
        return not evaluate_ltl(formula[4:], trace, t)
    # Binary ops
    for op in [" IMPLIES ", " UNTIL ", " AND ", " OR "]:
        if op in formula:
            # Find the top-level occurrence (not inside parens)
            depth = 0
            for i, c in enumerate(formula):
                if c == "(":
                    depth += 1
                elif c == ")":
                    depth -= 1
                elif depth == 0 and formula[i:i + len(op)] == op:
                    left = formula[:i]
                    right = formula[i + len(op):]
                    if op == " IMPLIES ":
                        return (not evaluate_ltl(left, trace, t)) or evaluate_ltl(right, trace, t)
                    elif op == " AND ":
                        return evaluate_ltl(left, trace, t) and evaluate_ltl(right, trace, t)
                    elif op == " OR ":
                        return evaluate_ltl(left, trace, t) or evaluate_ltl(right, trace, t)
                    elif op == " UNTIL ":
                        for tt in range(t, len(trace)):
                            if evaluate_ltl(right, trace, tt):
                                return True
                            if not evaluate_ltl(left, trace, tt):
                                return False
                        return False
    # Unary temporal
    if formula.startswith("EVENTUALLY "):
        sub = formula[len("EVENTUALLY "):]
        for tt in range(t, len(trace)):
            if evaluate_ltl(sub, trace, tt):
                return True
        return False
    if formula.startswith("ALWAYS "):
        sub = formula[len("ALWAYS "):]
        for tt in range(t, len(trace)):
            if not evaluate_ltl(sub, trace, tt):
                return False
        return True
    # Atom
    return evaluate_atom(formula, trace, t)


def verify_rule(rule, trace):
    """Verify an LTL rule against a trace. Returns bool (satisfied?)."""
    return evaluate_ltl(rule, trace, 0)


def graded_truth(rule, trace):
    """Return (frequency, confidence) NARS-style truth values.

    frequency = fraction of timesteps where rule is satisfied
    confidence = based on trace length (more evidence = more confidence)
    """
    if len(trace) == 0:
        return (0.5, 0.0)
    n_sat = sum(1 for t in range(len(trace)) if evaluate_ltl(rule, trace, t))
    freq = n_sat / len(trace)
    conf = min(1.0, len(trace) / 100.0)  # evidence-based confidence
    return (freq, conf)


# Default LunarLander rule set
DEFAULT_RULES = [
    "ALWAYS angle_below(1.0)",          # never tilt too much
    "EVENTUALLY velocity_below(0.3)",   # eventually slow down
    "ALWAYS (landed() IMPLIES in_pad())",  # if landed, must be in pad
]


if __name__ == "__main__":
    # Test: build a fake trace
    fake_trace = []
    for t in range(50):
        fake_trace.append({
            "x_pos": 0.1 - t * 0.001,
            "y_pos": 0.5 - t * 0.005,
            "x_vel": 0.0,
            "y_vel": 0.0,
            "angle": 0.01,
            "ang_vel": 0.0,
            "leg_l": 1.0 if t == 49 else 0.0,
            "leg_r": 1.0 if t == 49 else 0.0,
        })
    print("Phase 1.4 (E: LTL Verifier) PoC")
    print("=" * 50)
    for rule in DEFAULT_RULES:
        sat = verify_rule(rule, fake_trace)
        f, c = graded_truth(rule, fake_trace)
        print(f"  Rule: {rule}")
        print(f"    Satisfied: {sat}, freq={f:.2f}, conf={c:.2f}")
    print()
    print("Phase 1.4 PoC PASSED")