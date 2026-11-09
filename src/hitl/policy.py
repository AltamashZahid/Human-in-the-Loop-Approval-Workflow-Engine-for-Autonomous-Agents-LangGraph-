"""Risk-routing policy: a small YAML DSL, evaluated top to bottom, first match wins.

    default:                      # used when no rule matches
      route: approval
      escalation: [{role: ops, timeout_s: 60}]
    rules:
      - name: small_refund
        when: {tool: refund, args.amount: {lte: 50}}
        route: auto               # auto | approval | deny
      - name: large_refund
        when: {tool: refund, args.amount: {gt: 50}}
        route: approval
        escalation:               # each tier times out into the next; the last one rejects
          - {role: support_lead, timeout_s: 60}
          - {role: finance_manager, timeout_s: 300}

The policy fails closed. A missing field or a value of the wrong type makes a condition false,
so a malformed call cannot match an `auto` rule and falls through to the default. A policy with
no default, or a default that auto-approves, is refused when it is loaded.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

ROUTES = {"auto", "approval", "deny"}
_MISSING = object()
_ORDERED = {"lt": operator.lt, "lte": operator.le, "gt": operator.gt, "gte": operator.ge}


class PolicyError(ValueError):
    pass


@dataclass(frozen=True)
class Tier:
    role: str
    timeout_s: float


@dataclass(frozen=True)
class Route:
    route: str
    rule: str
    escalation: tuple[Tier, ...] = ()


def _lookup(call: dict[str, Any], path: str) -> Any:
    value: Any = call
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return _MISSING
        value = value[part]
    return value


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _test(op: str, value: Any, expected: Any) -> bool:
    if op == "eq":
        return value == expected
    if op == "ne":
        return value != expected
    if op == "in":
        return value in expected
    if op == "not_in":
        return value not in expected
    return _is_number(value) and _ORDERED[op](value, expected)


def _check_condition(rule: str, field: str, condition: Any) -> None:
    if not isinstance(condition, dict):
        return
    for op, expected in condition.items():
        if op in _ORDERED:
            if not _is_number(expected):
                raise PolicyError(f"{rule}: {field}.{op} needs a number, got {expected!r}")
        elif op in ("in", "not_in"):
            if not isinstance(expected, list):
                raise PolicyError(f"{rule}: {field}.{op} needs a list, got {expected!r}")
        elif op not in ("eq", "ne"):
            raise PolicyError(f"{rule}: unknown operator {op!r} on {field}")


def _parse_route(name: str, spec: dict[str, Any]) -> Route:
    route = spec.get("route")
    if route not in ROUTES:
        raise PolicyError(f"{name}: route must be one of {sorted(ROUTES)}, got {route!r}")
    tiers = tuple(
        Tier(role=str(tier["role"]), timeout_s=float(tier["timeout_s"]))
        for tier in spec.get("escalation", [])
    )
    if route == "approval" and not tiers:
        raise PolicyError(f"{name}: an approval route needs at least one escalation tier")
    if any(tier.timeout_s <= 0 for tier in tiers):
        raise PolicyError(f"{name}: timeouts must be positive")
    return Route(route=route, rule=name, escalation=tiers)


class Policy:
    def __init__(self, spec: dict[str, Any]) -> None:
        if not isinstance(spec.get("default"), dict):
            raise PolicyError("the policy needs a default route")
        self.default = _parse_route("default", spec["default"])
        if self.default.route == "auto":
            raise PolicyError("the default route must not be auto: the policy has to fail closed")
        self.rules: list[tuple[dict[str, Any], Route]] = []
        seen: set[str] = set()
        for rule in spec.get("rules", []):
            name = rule.get("name")
            if not name or name in seen:
                raise PolicyError(f"rule names must be present and unique, got {name!r}")
            seen.add(name)
            when = rule.get("when")
            if not isinstance(when, dict) or not when:
                raise PolicyError(f"{name}: a rule needs a non-empty when")
            for field, condition in when.items():
                _check_condition(name, field, condition)
            self.rules.append((when, _parse_route(name, rule)))

    @classmethod
    def load(cls, path: str | Path) -> Policy:
        return cls(yaml.safe_load(Path(path).read_text(encoding="utf-8")))

    def evaluate(self, tool: str, args: dict[str, Any]) -> Route:
        call = {"tool": tool, "args": args}
        for when, route in self.rules:
            if all(self._matches(call, field, condition) for field, condition in when.items()):
                return route
        return self.default

    @staticmethod
    def _matches(call: dict[str, Any], field: str, condition: Any) -> bool:
        value = _lookup(call, field)
        if value is _MISSING:
            return False
        if isinstance(condition, dict):
            return all(_test(op, value, expected) for op, expected in condition.items())
        return value == condition
