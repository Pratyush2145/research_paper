"""Adversarial robustness and evasion testing package.

Implements approved transformation taxonomy (Tier A, Tier B, Tier C),
paired contingency evaluation, statistical significance testing,
and diagnostic robustness audits.
"""

from src.adversarial.attacks import ATTACK_REGISTRY, AttackResult, apply_attack
from src.adversarial.evaluator import AdversarialEvaluator

__all__ = [
    "ATTACK_REGISTRY",
    "AttackResult",
    "apply_attack",
    "AdversarialEvaluator",
]

