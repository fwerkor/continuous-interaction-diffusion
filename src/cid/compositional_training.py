from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter, defaultdict, deque
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cid.data import TrajectoryExample, trajectory_to_dict
from cid.distill import (
    TeacherCellPlan,
    TeacherFrame,
    TeacherPlan,
    TeacherScheduleConfig,
    TeacherTask,
    compile_teacher_plans,
    review_teacher_plans,
)
from cid.grounding import Anchor, AnchorKind, CognitiveLink, LinkRelation, ObjectRef
from cid.state import CellLifecycle, CognitiveRole

COMPOSITIONAL_FAMILIES = (
    "boolean_dag",
    "blocked_reachability",
    "spatial_intervention",
    "causal_intervention",
    "ordering_mesh",
    "quantifier_dag",
    "candidate_elimination",
    "numeric_dag",
    "iterative_policy",
    "internal_hypothesis_repair",
)

TRAIN_CAPACITY_COUNTS = {8: 6000, 16: 5000, 32: 4000, 64: 3000, 128: 2000}
PROBE_CAPACITY_COUNTS = {16: 1000, 32: 1000, 64: 1000, 128: 1000}
TRAIN_DOMAINS = (
    "software release",
    "warehouse routing",
    "laboratory workflow",
    "course planning",
    "manufacturing line",
    "incident response",
    "research review",
    "network operations",
    "supply chain",
    "robot fleet",
    "ecology survey",
    "data pipeline",
    "clinic scheduling",
    "quality assurance",
    "satellite telemetry",
    "conference logistics",
)
PROBE_DOMAINS = (
    "aerospace certification",
    "maritime navigation",
    "archaeological fieldwork",
    "pharmacology protocol",
    "power-grid restoration",
    "legal case triage",
    "semiconductor fabrication",
    "polar expedition",
)
SEMANTIC_TEXT_CAP = 144


@dataclass(frozen=True, slots=True)
class CompositionalTrainingConfig:
    seed: int = 20260813
    variants_per_task: int = 2
    probe_variants_per_task: int = 1
    train_capacity_counts: tuple[tuple[int, int], ...] = tuple(TRAIN_CAPACITY_COUNTS.items())
    probe_capacity_counts: tuple[tuple[int, int], ...] = tuple(PROBE_CAPACITY_COUNTS.items())

    def __post_init__(self) -> None:
        for capacity, count in (*self.train_capacity_counts, *self.probe_capacity_counts):
            if capacity not in {8, 16, 32, 64, 128}:
                raise ValueError(f"unsupported compositional capacity bucket: {capacity}")
            if count <= 0 or count % len(COMPOSITIONAL_FAMILIES):
                raise ValueError("capacity counts must be positive and divisible by family count")
        if self.variants_per_task <= 0 or self.probe_variants_per_task <= 0:
            raise ValueError("trajectory variant counts must be positive")


@dataclass(frozen=True, slots=True)
class ReasoningNode:
    cell_id: str
    semantic_text: str
    parents: tuple[str, ...] = ("premises",)
    roles: tuple[CognitiveRole, ...] = (CognitiveRole.PERCEPT, CognitiveRole.PLAN)


@dataclass(frozen=True, slots=True)
class GeneratedCase:
    task: TeacherTask
    nodes: tuple[ReasoningNode, ...]
    initial_text: str
    initial_hypothesis: str
    final_text: str
    correction: bool = False


_GREEK = (
    "amber",
    "birch",
    "cobalt",
    "delta",
    "ember",
    "frost",
    "garnet",
    "harbor",
    "indigo",
    "juniper",
    "kelp",
    "lumen",
    "marble",
    "nimbus",
    "onyx",
    "pearl",
    "quartz",
    "raven",
    "saffron",
    "tundra",
    "umber",
    "violet",
    "willow",
    "xenon",
    "yarrow",
    "zephyr",
    "atlas",
    "boreal",
    "cirrus",
    "dune",
    "echo",
    "fjord",
)
_NAMES = (
    "Ari",
    "Bea",
    "Cato",
    "Dara",
    "Eli",
    "Faye",
    "Gus",
    "Hana",
    "Ivo",
    "Jia",
    "Kian",
    "Lina",
    "Miro",
    "Nia",
    "Oren",
    "Pia",
    "Quin",
    "Ravi",
    "Sora",
    "Tavi",
    "Uma",
    "Vera",
    "Wren",
    "Xavi",
    "Yuna",
    "Zane",
    "Bram",
    "Cyra",
    "Davi",
    "Esme",
)
_SCOPE_TEXTS = (
    "Use only stated constraints; preserve rule direction, exceptions, and counterfactual scope.",
    "Respect the stated semantics exactly; do not import converses, defaults, or unstated assumptions.",
    "Track licensed dependencies only; keep distractors separate and apply local overrides exactly once.",
    "Preserve direction, scope, explicit negation, and intervention semantics while composing dependencies.",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _short(text: str) -> str:
    normalized = " ".join(str(text).split())
    return normalized if len(normalized) <= 140 else normalized[:137].rstrip() + "..."


def _answer_anchor(answer: str) -> Anchor:
    digest = hashlib.sha256(answer.casefold().encode()).hexdigest()[:16]
    return Anchor(
        anchor_id=f"text:{digest}",
        kind=AnchorKind.TEXT,
        value=answer,
        confidence=1.0,
    )


def _cell(
    cell_id: str,
    text: str,
    roles: dict[CognitiveRole, float],
    uncertainty: float,
    noise: float,
    lifecycle: CellLifecycle,
    *,
    anchors: tuple[Anchor, ...] = (),
    links: tuple[CognitiveLink, ...] = (),
) -> TeacherCellPlan:
    return TeacherCellPlan(
        cell_id=cell_id,
        semantic_text=_short(text),
        roles=roles,
        uncertainty=uncertainty,
        noise=noise,
        lifecycle=lifecycle,
        anchors=anchors,
        links=links,
    )


def _capacity_target(capacity: int, rng: random.Random, *, probe: bool) -> int:
    ranges = {
        8: (6, 8),
        16: (11, 16),
        32: (22, 32),
        64: (44, 64),
        128: (76, 120),
    }
    lo, hi = ranges[capacity]
    if probe:
        lo = max(lo, int(hi * 0.9))
    return rng.randint(lo, hi)


def _refinement_count(capacity: int) -> int:
    return {8: 2, 16: 3, 32: 4, 64: 5, 128: 6}[capacity]


def _surface_style(capacity: int, rng: random.Random, probe: bool) -> str:
    if capacity >= 64:
        return rng.choice(("compact", "compact", "ledger"))
    if probe:
        return rng.choice(("narrative", "ledger", "compact"))
    return rng.choice(("narrative", "narrative", "compact", "ledger"))


def _render_records(
    style: str,
    compact: Iterable[str],
    narrative: Iterable[str],
) -> str:
    compact_items = tuple(compact)
    narrative_items = tuple(narrative)
    if len(compact_items) != len(narrative_items):
        raise ValueError("compact and narrative record renderings must align")
    if style == "narrative":
        return "; ".join(narrative_items)
    if style == "ledger":
        return " | ".join(f"R{i + 1}:{item}" for i, item in enumerate(compact_items))
    if style == "compact":
        return ",".join(compact_items)
    raise ValueError(f"unsupported surface style: {style}")


def _labels(prefix: str, count: int, rng: random.Random, *, natural_limit: int = 20) -> list[str]:
    if count <= natural_limit:
        pool = list(_GREEK if prefix != "person" else _NAMES)
        if count <= len(pool):
            return rng.sample(pool, count)
    stem = prefix[0].upper()
    width = max(2, len(str(count)))
    offset = rng.randrange(0, 1000)
    return [f"{stem}{offset + i:0{width}d}" for i in range(count)]


def _role_weights(roles: tuple[CognitiveRole, ...]) -> dict[CognitiveRole, float]:
    if not roles:
        return {CognitiveRole.PERCEPT: 1.0}
    first, *rest = roles
    result = {first: 0.9}
    for role in rest:
        result[role] = 0.55
    return result


def _plan_for(case: GeneratedCase, rng: random.Random) -> TeacherPlan:
    capacity = int(case.task.metadata["thought_capacity_bucket"])
    refine_count = _refinement_count(capacity)
    scope = rng.choice(_SCOPE_TEXTS)
    nodes = case.nodes
    if len(nodes) + 4 > capacity:
        raise ValueError(f"case {case.task.task_id} exceeds its capacity bucket")

    boundaries = [
        math.floor((index + 1) * len(nodes) / refine_count) for index in range(refine_count)
    ]

    def base(
        hypothesis: str, uncertainty: float, lifecycle: CellLifecycle
    ) -> list[TeacherCellPlan]:
        return [
            _cell(
                "scope",
                scope,
                {CognitiveRole.CONSTRAINT: 1.0, CognitiveRole.PLAN: 0.3},
                0.04,
                0.02,
                CellLifecycle.STABLE,
                links=(CognitiveLink(LinkRelation.CONSTRAINS, ObjectRef.cell("hypothesis"), 0.95),),
            ),
            _cell(
                "premises",
                case.initial_text,
                {CognitiveRole.PERCEPT: 0.75, CognitiveRole.CONSTRAINT: 0.78},
                0.10,
                0.06,
                CellLifecycle.STABLE,
                links=(CognitiveLink(LinkRelation.CONSTRAINS, ObjectRef.cell("hypothesis"), 0.95),),
            ),
            _cell(
                "hypothesis",
                hypothesis,
                {CognitiveRole.HYPOTHESIS: 1.0, CognitiveRole.PLAN: 0.25},
                uncertainty,
                max(0.03, uncertainty * 0.7),
                lifecycle,
                links=(CognitiveLink(LinkRelation.DEPENDS_ON, ObjectRef.cell("premises"), 1.0),),
            ),
        ]

    frames: list[TeacherFrame] = [
        TeacherFrame(
            phase="initial",
            display="Reasoning.",
            cells=tuple(
                base(
                    case.initial_hypothesis, 0.82 if case.correction else 0.74, CellLifecycle.ACTIVE
                )
            ),
        )
    ]
    introduced: list[ReasoningNode] = []
    for frame_index, boundary in enumerate(boundaries):
        introduced = list(nodes[:boundary])
        if case.correction and frame_index == 0:
            hypothesis = "The tempting shortcut is plausible but remains unchecked against the full constraint graph."
            uncertainty = 0.72
        elif case.correction and frame_index == 1:
            hypothesis = "The shortcut conflicts with an authoritative dependency; reopen only that provisional branch."
            uncertainty = 0.58
        elif case.correction:
            hypothesis = "The conflicting shortcut is discarded; the constraint-consistent branch is now preferred."
            uncertainty = max(0.06, 0.30 - 0.05 * frame_index)
        else:
            hypothesis = (
                "Several query-relevant branches remain unresolved; preserve them until their merge conditions are checked."
                if frame_index + 1 < refine_count
                else "All query-relevant branches have been composed; the candidate conclusion is stable."
            )
            uncertainty = max(0.05, 0.48 - 0.07 * frame_index)
        node_cells = []
        for node in introduced:
            links = tuple(
                CognitiveLink(LinkRelation.DERIVED_FROM, ObjectRef.cell(parent), 1.0)
                for parent in node.parents[:8]
            )
            node_cells.append(
                _cell(
                    node.cell_id,
                    node.semantic_text,
                    _role_weights(node.roles),
                    max(0.04, 0.20 - 0.02 * frame_index),
                    max(0.02, 0.12 - 0.015 * frame_index),
                    CellLifecycle.ACTIVE
                    if frame_index + 1 < refine_count
                    else CellLifecycle.STABLE,
                    links=links,
                )
            )
        frames.append(
            TeacherFrame(
                phase=f"refine:{frame_index}",
                display="Reasoning.",
                cells=tuple(
                    base(
                        hypothesis,
                        uncertainty,
                        CellLifecycle.ACTIVE
                        if frame_index + 1 < refine_count
                        else CellLifecycle.STABLE,
                    )
                    + node_cells
                ),
            )
        )

    final_nodes = []
    for node in nodes:
        final_nodes.append(
            _cell(
                node.cell_id,
                node.semantic_text,
                _role_weights(node.roles),
                0.03,
                0.02,
                CellLifecycle.STABLE,
                links=tuple(
                    CognitiveLink(LinkRelation.DERIVED_FROM, ObjectRef.cell(parent), 1.0)
                    for parent in node.parents[:8]
                ),
            )
        )
    answer = str(case.task.reference_answer)
    answer_parent = nodes[-1].cell_id if nodes else "premises"
    final_cells = (
        base(
            "The checked dependency state supports the final answer.",
            0.025,
            CellLifecycle.STABLE,
        )
        + final_nodes
        + [
            _cell(
                "answer",
                case.final_text,
                {CognitiveRole.CONCLUSION: 1.0},
                0.015,
                0.01,
                CellLifecycle.STABLE,
                anchors=(_answer_anchor(answer),),
                links=(
                    CognitiveLink(LinkRelation.DERIVED_FROM, ObjectRef.cell(answer_parent), 1.0),
                ),
            )
        ]
    )
    frames.append(TeacherFrame("final", answer, tuple(final_cells)))
    return TeacherPlan(
        task_id=case.task.task_id, final_answer=answer, frames=tuple(frames), needs=()
    )


def _task(
    *,
    task_id: str,
    family: str,
    prompt: str,
    answer: str,
    spec: dict[str, Any],
    capacity: int,
    live_cells: int,
    domain: str,
    topology: str,
    primitives: tuple[str, ...],
    distractors: int,
    style: str,
    probe: bool,
    correction: bool = False,
) -> TeacherTask:
    return TeacherTask(
        task_id=task_id,
        prompt=prompt,
        metadata={
            "task_kind": "compositional_longtail_reasoning",
            "family": family,
            "interaction_pattern": "no_tool_internal_refinement",
            "training_mode": "no_tool_required",
            "generated_by": "GPT-5.6-Sol-authored-deterministic-compositional-v1",
            "domain": domain,
            "topology": topology,
            "reasoning_primitives": list(primitives),
            "composition_count": len(primitives),
            "distractor_count": distractors,
            "surface_style": style,
            "thought_capacity_bucket": capacity,
            "target_live_cells": live_cells,
            "internal_correction": correction,
            "generalization_split": "ood_probe" if probe else "train_longtail",
            "logic_spec": spec,
        },
        reference_answer=answer,
    )


def _bool_op(op: str, left: bool, right: bool) -> bool:
    if op == "and":
        return left and right
    if op == "or":
        return left or right
    if op == "xor":
        return left ^ right
    if op == "nand":
        return not (left and right)
    raise ValueError(op)


def _longest_parent_depth(parents: list[tuple[int, ...]]) -> int:
    depth: list[int] = []
    for item in parents:
        depth.append(1 + max((depth[parent] for parent in item), default=0))
    return max(depth, default=0)


def _boolean_dag_case(rng: random.Random, index: int, capacity: int, probe: bool) -> GeneratedCase:
    live = _capacity_target(capacity, rng, probe=probe)
    budget = live - 4
    domain = rng.choice(PROBE_DOMAINS if probe else TRAIN_DOMAINS)
    style = _surface_style(capacity, rng, probe)
    seed_labels = _labels("seed", 3, rng)
    node_labels = [f"g{i}" for i in range(budget)]
    values = [bool(rng.getrandbits(1)) for _ in seed_labels]
    ops = ("and", "or", "xor", "nand")
    rules: list[dict[str, Any]] = []
    parents_idx: list[tuple[int, ...]] = []
    nodes: list[ReasoningNode] = []
    for i, label in enumerate(node_labels):
        available = len(seed_labels) + i
        if probe and i > 4 and rng.random() < 0.45:
            a = rng.randrange(max(0, available - 12), available)
            b = rng.randrange(0, available)
        else:
            a = rng.randrange(max(0, available - 6), available)
            b = rng.randrange(0, available)
        if b == a:
            b = (b + 1) % available
        op = rng.choice(ops if probe or capacity >= 32 else ops[:3])
        value = _bool_op(op, values[a], values[b])
        values.append(value)
        rules.append({"op": op, "a": a, "b": b})
        derived_parents = tuple(
            parent - len(seed_labels) for parent in (a, b) if parent >= len(seed_labels)
        )
        parents_idx.append(derived_parents)
        parent_cells = tuple(f"r{parent}" for parent in derived_parents) or ("premises",)
        nodes.append(
            ReasoningNode(
                f"r{i}",
                f"{label}={'T' if value else 'F'} from {op.upper()} of its two declared parents.",
                parent_cells,
            )
        )
    statements: list[tuple[str, str]] = []
    all_labels = seed_labels + node_labels
    for i, rule in enumerate(rules):
        a, b = all_labels[rule["a"]], all_labels[rule["b"]]
        compact = f"{node_labels[i]}={rule['op'].upper()}({a},{b})"
        narrative = {
            "and": f"{node_labels[i]} is true exactly when both {a} and {b} are true",
            "or": f"{node_labels[i]} is true when at least one of {a} and {b} is true",
            "xor": f"{node_labels[i]} is true when exactly one of {a} and {b} is true",
            "nand": f"{node_labels[i]} is false only when both {a} and {b} are true",
        }[str(rule["op"])]
        statements.append((compact, narrative))
    distractors = max(1, budget // 12)
    for i in range(distractors):
        source = rng.choice(all_labels)
        statements.append(
            (
                f"note{i}=independent({source})",
                f"note {i} is independent of the query graph and merely mentions {source}",
            )
        )
    rng.shuffle(statements)
    intro = ", ".join(f"{label}={'T' if values[i] else 'F'}" for i, label in enumerate(seed_labels))
    rendered = _render_records(
        style,
        (item[0] for item in statements),
        (item[1] for item in statements),
    )
    prompt = (
        f"In a {domain} boolean dependency graph, seeds are {intro}. Evaluate rules exactly: "
        + rendered
        + f". What is {node_labels[-1]}? Answer exactly `true` or `false`."
    )
    answer = "true" if values[-1] else "false"
    depth = _longest_parent_depth(parents_idx)
    spec = {"kind": "boolean_dag", "seeds": values[:3], "rules": rules}
    task = _task(
        task_id=f"{'probe' if probe else 'comp'}-boolean-dag-c{capacity}-{index:06d}",
        family="boolean_dag",
        prompt=prompt,
        answer=answer,
        spec=spec,
        capacity=capacity,
        live_cells=live,
        domain=domain,
        topology="multi_merge_dag" if probe else "dag",
        primitives=("boolean_gates", "dependency_graph", "branch_merge"),
        distractors=distractors,
        style=style,
        probe=probe,
    )
    task.metadata["dependency_depth"] = depth
    return GeneratedCase(
        task,
        tuple(nodes),
        "Build the directed gate graph and keep independent notes outside the query ancestry.",
        "The target gate remains provisional until all parent branches are merged.",
        f"Final answer: {answer}.",
    )


def _blocked_reachability_case(
    rng: random.Random, index: int, capacity: int, probe: bool
) -> GeneratedCase:
    live = _capacity_target(capacity, rng, probe=probe)
    budget = live - 4
    domain = rng.choice(PROBE_DOMAINS if probe else TRAIN_DOMAINS)
    style = _surface_style(capacity, rng, probe)
    labels = _labels("node", budget + 3, rng, natural_limit=14)
    start, target = labels[0], labels[budget]
    edges: list[tuple[int, int]] = [(i, i + 1) for i in range(budget)]
    for _ in range(max(2, budget // 2)):
        a = rng.randrange(0, budget)
        b = rng.randrange(a + 1, min(budget + 1, a + 8))
        edges.append((a, b))
    blocked: set[int] = set()
    make_unreachable = index % 2 == 1
    if make_unreachable:
        cut = rng.randrange(max(1, budget // 3), max(2, 2 * budget // 3))
        blocked.add(cut)
        edges = [edge for edge in edges if not (edge[0] < cut < edge[1])]
    for _ in range(max(1, budget // 16)):
        candidate = rng.randrange(1, budget)
        if not make_unreachable or candidate != next(iter(blocked), -1):
            blocked.add(candidate)
    adjacency: dict[int, list[int]] = defaultdict(list)
    for a, b in edges:
        adjacency[a].append(b)
    queue = deque([0])
    seen = {0}
    while queue:
        cur = queue.popleft()
        for nxt in adjacency[cur]:
            if nxt in blocked or nxt in seen:
                continue
            seen.add(nxt)
            queue.append(nxt)
    answer = "reachable" if budget in seen else "unreachable"
    nodes = []
    for i in range(budget):
        vertex = i + 1
        state = (
            "blocked" if vertex in blocked else ("reachable" if vertex in seen else "not reached")
        )
        parent_candidates = [a for a, b in edges if b == vertex and a > 0]
        parents = tuple(f"r{a - 1}" for a in parent_candidates[:3]) or ("premises",)
        nodes.append(
            ReasoningNode(
                f"r{i}",
                f"{labels[vertex]} is {state} under the directed-edge and block constraints.",
                parents,
            )
        )
    edge_text = _render_records(
        style,
        (f"{labels[a]}>{labels[b]}" for a, b in edges),
        (f"there is a one-way route from {labels[a]} to {labels[b]}" for a, b in edges),
    )
    blocked_text = ",".join(labels[i] for i in sorted(blocked)) or "none"
    prompt = (
        f"For a {domain} route graph, edges are directed: {edge_text}. Blocked nodes: {blocked_text}. "
        f"Can {start} reach {target} without entering a blocked node? Answer exactly `reachable` or `unreachable`."
    )
    spec = {
        "kind": "blocked_reachability",
        "nodes": len(labels),
        "edges": edges,
        "blocked": sorted(blocked),
        "start": 0,
        "target": budget,
    }
    task = _task(
        task_id=f"{'probe' if probe else 'comp'}-reachability-c{capacity}-{index:06d}",
        family="blocked_reachability",
        prompt=prompt,
        answer=answer,
        spec=spec,
        capacity=capacity,
        live_cells=live,
        domain=domain,
        topology="cutset_mesh" if probe else "directed_graph",
        primitives=("reachability", "constraint_filtering", "cutset"),
        distractors=max(1, budget // 16),
        style=style,
        probe=probe,
    )
    task.metadata["dependency_depth"] = budget
    return GeneratedCase(
        task,
        tuple(nodes),
        "Track the legal frontier only; blocked vertices cannot transmit reachability.",
        "Reachability is provisional until every legal bypass around the blockers is checked.",
        f"Final answer: {answer}.",
    )


def _direction(dx: int, dy: int) -> str:
    if dx == 0 and dy == 0:
        return "same"
    ns = "north" if dy > 0 else ("south" if dy < 0 else "")
    ew = "east" if dx > 0 else ("west" if dx < 0 else "")
    return ns + ("-" if ns and ew else "") + ew


def _spatial_intervention_case(
    rng: random.Random, index: int, capacity: int, probe: bool
) -> GeneratedCase:
    live = _capacity_target(capacity, rng, probe=probe)
    budget = live - 4
    domain = rng.choice(PROBE_DOMAINS if probe else TRAIN_DOMAINS)
    style = _surface_style(capacity, rng, probe)
    directions = (("N", 0, 1), ("S", 0, -1), ("E", 1, 0), ("W", -1, 0))
    steps = [rng.choice(directions) for _ in range(budget)]
    changed = rng.randrange(max(1, budget // 4), max(2, 3 * budget // 4))
    replacement = rng.choice([item for item in directions if item[0] != steps[changed][0]])
    updated = list(steps)
    updated[changed] = replacement
    dx = dy = 0
    nodes = []
    for i, (_, sx, sy) in enumerate(updated):
        dx += sx
        dy += sy
        parent = (f"r{i - 1}",) if i else ("premises",)
        marker = " after the local replacement" if i == changed else ""
        nodes.append(
            ReasoningNode(f"r{i}", f"Prefix {i + 1}: displacement=({dx},{dy}){marker}.", parent)
        )
    answer = _direction(dx, dy)
    direction_words = {"N": "north", "S": "south", "E": "east", "W": "west"}
    chain = _render_records(
        style,
        (step[0] for step in steps),
        (f"move one unit {direction_words[step[0]]}" for step in steps),
    )
    prompt = (
        f"A {domain} path starts at (0,0) with ordered unit moves [{chain}]. Counterfactually replace only move "
        f"{changed + 1} by {direction_words[replacement[0]]}; keep all other moves unchanged. Where is the endpoint relative to the start? "
        "Answer one of north, south, east, west, north-east, north-west, south-east, south-west, same."
    )
    spec = {
        "kind": "spatial_intervention",
        "steps": [item[0] for item in steps],
        "changed": changed,
        "replacement": replacement[0],
    }
    task = _task(
        task_id=f"{'probe' if probe else 'comp'}-spatial-c{capacity}-{index:06d}",
        family="spatial_intervention",
        prompt=prompt,
        answer=answer,
        spec=spec,
        capacity=capacity,
        live_cells=live,
        domain=domain,
        topology="long_chain",
        primitives=("spatial_composition", "counterfactual", "local_recompute"),
        distractors=0,
        style=style,
        probe=probe,
    )
    task.metadata["dependency_depth"] = budget
    return GeneratedCase(
        task,
        tuple(nodes),
        "Preserve the unchanged path prefix and replace exactly one directed move.",
        "The endpoint is provisional until the modified suffix is recomposed.",
        f"Final direction: {answer}.",
    )


def _causal_intervention_case(
    rng: random.Random, index: int, capacity: int, probe: bool
) -> GeneratedCase:
    live = _capacity_target(capacity, rng, probe=probe)
    budget = live - 4
    domain = rng.choice(PROBE_DOMAINS if probe else TRAIN_DOMAINS)
    style = _surface_style(capacity, rng, probe)
    seeds = [bool(rng.getrandbits(1)) for _ in range(3)]
    values = list(seeds)
    rules: list[dict[str, Any]] = []
    ops = ("and", "or", "xor")
    for i in range(budget):
        available = 3 + i
        a = rng.randrange(max(0, available - 7), available)
        b = rng.randrange(0, available)
        if a == b:
            b = (b + 1) % available
        op = rng.choice(ops)
        values.append(_bool_op(op, values[a], values[b]))
        rules.append({"a": a, "b": b, "op": op})
    intervention_node = 3 + rng.randrange(max(1, budget // 4), max(2, 3 * budget // 4))
    intervention_value = not values[intervention_node]
    recomputed = list(seeds)
    nodes = []
    for i, rule in enumerate(rules):
        node_index = 3 + i
        if node_index == intervention_node:
            value = intervention_value
        else:
            value = _bool_op(rule["op"], recomputed[rule["a"]], recomputed[rule["b"]])
        recomputed.append(value)
        parents = (
            ("premises",)
            if node_index == intervention_node
            else tuple(f"r{parent - 3}" for parent in (rule["a"], rule["b"]) if parent >= 3)
            or ("premises",)
        )
        note = " fixed by do-intervention" if node_index == intervention_node else ""
        nodes.append(ReasoningNode(f"r{i}", f"c{i}={'T' if value else 'F'}{note}.", parents))
    answer = "true" if recomputed[-1] else "false"

    def causal_ref(value: int) -> str:
        return f"s{value}" if value < 3 else f"c{value - 3}"

    rule_text = _render_records(
        style,
        (
            f"c{i}={rule['op'].upper()}({causal_ref(int(rule['a']))},{causal_ref(int(rule['b']))})"
            for i, rule in enumerate(rules)
        ),
        (
            f"c{i} is the {str(rule['op']).upper()} of {causal_ref(int(rule['a']))} and {causal_ref(int(rule['b']))}"
            for i, rule in enumerate(rules)
        ),
    )
    seed_text = ",".join(f"s{i}={'T' if value else 'F'}" for i, value in enumerate(seeds))
    prompt = (
        f"In a {domain} causal DAG, {seed_text}; equations: {rule_text}. Apply do(c{intervention_node - 3}="
        f"{'T' if intervention_value else 'F'}), which replaces that node's equation and leaves all others unchanged. "
        f"What is c{budget - 1}? Answer exactly `true` or `false`."
    )
    spec = {
        "kind": "causal_intervention",
        "seeds": seeds,
        "rules": rules,
        "intervention": intervention_node,
        "value": intervention_value,
    }
    task = _task(
        task_id=f"{'probe' if probe else 'comp'}-causal-c{capacity}-{index:06d}",
        family="causal_intervention",
        prompt=prompt,
        answer=answer,
        spec=spec,
        capacity=capacity,
        live_cells=live,
        domain=domain,
        topology="intervened_dag",
        primitives=("causal_dag", "intervention", "descendant_recompute"),
        distractors=0,
        style=style,
        probe=probe,
    )
    task.metadata["dependency_depth"] = budget
    return GeneratedCase(
        task,
        tuple(nodes),
        "Cut incoming causes at the intervention node; recompute only through the remaining structural equations.",
        "The post-intervention target remains provisional until all affected descendants are recomputed.",
        f"Final answer: {answer}.",
    )


def _ordering_mesh_case(
    rng: random.Random, index: int, capacity: int, probe: bool
) -> GeneratedCase:
    live = _capacity_target(capacity, rng, probe=probe)
    budget = live - 4
    domain = rng.choice(PROBE_DOMAINS if probe else TRAIN_DOMAINS)
    style = _surface_style(capacity, rng, probe)
    count = budget + 1
    order = _labels("person", count, rng, natural_limit=16)
    constraints = [(order[i], order[i + 1]) for i in range(count - 1)]
    extras = []
    for _ in range(max(2, budget // 5)):
        a = rng.randrange(0, count - 2)
        b = rng.randrange(a + 2, count)
        extras.append((order[a], order[b]))
    all_constraints = constraints + extras
    rng.shuffle(all_constraints)
    position = rng.randrange(1, count + 1)
    answer = order[position - 1]
    nodes = []
    for i in range(budget):
        parent = (f"r{i - 1}",) if i else ("premises",)
        nodes.append(
            ReasoningNode(
                f"r{i}",
                f"Order prefix {i + 2}: {order[i]} < {order[i + 1]} is fixed by the adjacency chain.",
                parent,
            )
        )
    rendered_constraints = _render_records(
        style,
        (f"{a}<{b}" for a, b in all_constraints),
        (f"{a} must occur before {b}" for a, b in all_constraints),
    )
    prompt = (
        f"In a {domain} schedule, all {count} items occupy distinct positions. Constraints: "
        + rendered_constraints
        + f". The adjacent constraints determine one total order. Which item is in position {position}? Answer with the item only."
    )
    spec = {"kind": "ordering_mesh", "order": order, "position": position}
    task = _task(
        task_id=f"{'probe' if probe else 'comp'}-ordering-c{capacity}-{index:06d}",
        family="ordering_mesh",
        prompt=prompt,
        answer=answer,
        spec=spec,
        capacity=capacity,
        live_cells=live,
        domain=domain,
        topology="constraint_mesh",
        primitives=("ordering", "transitive_closure", "redundancy_filtering"),
        distractors=len(extras),
        style=style,
        probe=probe,
    )
    task.metadata["dependency_depth"] = budget
    return GeneratedCase(
        task,
        tuple(nodes),
        "Recover the unique adjacency backbone; redundant transitive constraints must not change the order.",
        "The requested position remains provisional until the full order is reconstructed.",
        f"Final item: {answer}.",
    )


def _quantifier_dag_case(
    rng: random.Random, index: int, capacity: int, probe: bool
) -> GeneratedCase:
    live = _capacity_target(capacity, rng, probe=probe)
    budget = live - 4
    domain = rng.choice(PROBE_DOMAINS if probe else TRAIN_DOMAINS)
    style = _surface_style(capacity, rng, probe)
    classes = _labels("class", budget + 4, rng, natural_limit=12)
    inclusions: list[tuple[int, int]] = [(i, i + 1) for i in range(budget)]
    for _ in range(max(1, budget // 4)):
        a = rng.randrange(0, budget)
        upper = min(budget + 1, a + 8)
        if a + 1 < upper:
            b = rng.randrange(a + 1, upper)
            inclusions.append((a, b))
    disjoint = [(budget, budget + 1)]
    mode = index % 3
    query = budget if mode == 0 else (budget + 1 if mode == 1 else budget + 2)
    answer = ("entailed", "contradicted", "unknown")[mode]
    nodes = []
    for i in range(budget):
        parent = (f"r{i - 1}",) if i else ("premises",)
        nodes.append(
            ReasoningNode(
                f"r{i}",
                f"Membership closure reaches {classes[i + 1]} without using any converse.",
                parent,
            )
        )
    incl_text = _render_records(
        style,
        (f"{classes[a]}⊆{classes[b]}" for a, b in inclusions),
        (f"every {classes[a]} object is also {classes[b]}" for a, b in inclusions),
    )
    disjoint_text = _render_records(
        style,
        (f"{classes[a]}⊥{classes[b]}" for a, b in disjoint),
        (f"nothing can be both {classes[a]} and {classes[b]}" for a, b in disjoint),
    )
    prompt = (
        f"Use open-world class logic in a {domain} taxonomy. Object x belongs to {classes[0]}. Inclusions: {incl_text}. "
        f"Disjointness: {disjoint_text}. Is x in {classes[query]} entailed, contradicted, or unknown? Answer exactly one label."
    )
    spec = {
        "kind": "quantifier_dag",
        "classes": len(classes),
        "inclusions": inclusions,
        "disjoint": disjoint,
        "root": 0,
        "query": query,
    }
    task = _task(
        task_id=f"{'probe' if probe else 'comp'}-quantifier-c{capacity}-{index:06d}",
        family="quantifier_dag",
        prompt=prompt,
        answer=answer,
        spec=spec,
        capacity=capacity,
        live_cells=live,
        domain=domain,
        topology="taxonomy_dag",
        primitives=("set_inclusion", "open_world", "disjointness"),
        distractors=max(1, budget // 4),
        style=style,
        probe=probe,
    )
    task.metadata["dependency_depth"] = budget
    return GeneratedCase(
        task,
        tuple(nodes),
        "Propagate membership forward through inclusions; do not use converse membership or closed-world negation.",
        "The query label stays provisional until both positive closure and disjointness are checked.",
        f"Final label: {answer}.",
    )


def _candidate_elimination_case(
    rng: random.Random, index: int, capacity: int, probe: bool
) -> GeneratedCase:
    live = _capacity_target(capacity, rng, probe=probe)
    budget = live - 4
    domain = rng.choice(PROBE_DOMAINS if probe else TRAIN_DOMAINS)
    style = _surface_style(capacity, rng, probe)
    feature_count = 8 if capacity <= 16 else (12 if capacity <= 64 else 16)
    candidate_count = max(4, budget)
    candidates = [f"K{i:03d}" for i in range(candidate_count)]
    signatures: list[str] = []
    seen: set[str] = set()
    while len(signatures) < candidate_count:
        signature = "".join("1" if rng.getrandbits(1) else "0" for _ in range(feature_count))
        if signature not in seen:
            seen.add(signature)
            signatures.append(signature)
    winner = rng.randrange(candidate_count)
    wanted = signatures[winner]
    selected_features = list(range(feature_count))
    rng.shuffle(selected_features)
    selected_features = selected_features[: max(4, feature_count - 2)]
    survivors = [
        i for i, sig in enumerate(signatures) if all(sig[f] == wanted[f] for f in selected_features)
    ]
    if len(survivors) != 1:
        selected_features = list(range(feature_count))
    answer = candidates[winner]
    nodes = []
    for i in range(budget):
        candidate_index = i % candidate_count
        if candidate_index == winner:
            text = f"{candidates[candidate_index]} remains compatible with every required feature checked so far."
            roles = (CognitiveRole.HYPOTHESIS, CognitiveRole.PLAN)
        else:
            mismatch = next(
                (f for f in selected_features if signatures[candidate_index][f] != wanted[f]),
                selected_features[0],
            )
            text = f"{candidates[candidate_index]} is eliminated by feature f{mismatch}={wanted[mismatch]}."
            roles = (CognitiveRole.HYPOTHESIS, CognitiveRole.CONSTRAINT)
        nodes.append(ReasoningNode(f"r{i}", text, ("premises",), roles))
    table = _render_records(
        style,
        (f"{name}:{signature}" for name, signature in zip(candidates, signatures, strict=True)),
        (
            f"candidate {name} has feature vector {signature}"
            for name, signature in zip(candidates, signatures, strict=True)
        ),
    )
    requirements = _render_records(
        style,
        (f"f{f}={wanted[f]}" for f in selected_features),
        (f"feature f{f} must equal {wanted[f]}" for f in selected_features),
    )
    prompt = (
        f"In a {domain} candidate table, each bit string gives f0..f{feature_count - 1}. Table: {table}. "
        f"Requirements: {requirements}. Exactly one candidate satisfies all requirements. Which one? Answer with its ID only."
    )
    spec = {
        "kind": "candidate_elimination",
        "signatures": signatures,
        "features": selected_features,
        "wanted": wanted,
        "candidates": candidates,
    }
    task = _task(
        task_id=f"{'probe' if probe else 'comp'}-candidate-c{capacity}-{index:06d}",
        family="candidate_elimination",
        prompt=prompt,
        answer=answer,
        spec=spec,
        capacity=capacity,
        live_cells=live,
        domain=domain,
        topology="competing_hypotheses",
        primitives=("candidate_search", "constraint_intersection", "elimination"),
        distractors=candidate_count - 1,
        style=style,
        probe=probe,
    )
    task.metadata["dependency_depth"] = len(selected_features)
    task.metadata["reasoning_width"] = candidate_count
    return GeneratedCase(
        task,
        tuple(nodes),
        "Maintain competing candidate hypotheses and eliminate only on explicit feature mismatches.",
        "Several candidates remain plausible until all required feature intersections are applied.",
        f"Final candidate: {answer}.",
    )


def _numeric_dag_case(rng: random.Random, index: int, capacity: int, probe: bool) -> GeneratedCase:
    live = _capacity_target(capacity, rng, probe=probe)
    budget = live - 4
    domain = rng.choice(PROBE_DOMAINS if probe else TRAIN_DOMAINS)
    style = _surface_style(capacity, rng, probe)
    seeds = [rng.randint(-20, 20) for _ in range(3)]
    values = list(seeds)
    rules: list[dict[str, Any]] = []
    nodes = []
    ops = ("add", "sub", "max", "min")
    for i in range(budget):
        available = 3 + i
        a = rng.randrange(max(0, available - 8), available)
        b = rng.randrange(0, available)
        if a == b:
            b = (b + 1) % available
        op = rng.choice(ops)
        va, vb = values[a], values[b]
        value = {"add": va + vb, "sub": va - vb, "max": max(va, vb), "min": min(va, vb)}[op]
        value = max(-9999, min(9999, value))
        values.append(value)
        rules.append({"a": a, "b": b, "op": op})
        parent_cells = tuple(f"r{parent - 3}" for parent in (a, b) if parent >= 3) or ("premises",)
        nodes.append(
            ReasoningNode(
                f"r{i}",
                f"v{i}={value} after exact {op} merge of its declared parents.",
                parent_cells,
            )
        )
    answer = str(values[-1])

    def ref(idx: int) -> str:
        return f"s{idx}" if idx < 3 else f"v{idx - 3}"

    rule_text = _render_records(
        style,
        (f"v{i}={rule['op']}({ref(rule['a'])},{ref(rule['b'])})" for i, rule in enumerate(rules)),
        (
            f"v{i} is the {rule['op']} of {ref(rule['a'])} and {ref(rule['b'])}"
            for i, rule in enumerate(rules)
        ),
    )
    prompt = (
        f"In a {domain} integer dependency graph, seeds are s0={seeds[0]},s1={seeds[1]},s2={seeds[2]}. "
        f"Compute exactly: {rule_text}. What is v{budget - 1}? Answer with the integer only."
    )
    spec = {"kind": "numeric_dag", "seeds": seeds, "rules": rules}
    task = _task(
        task_id=f"{'probe' if probe else 'comp'}-numeric-c{capacity}-{index:06d}",
        family="numeric_dag",
        prompt=prompt,
        answer=answer,
        spec=spec,
        capacity=capacity,
        live_cells=live,
        domain=domain,
        topology="numeric_merge_dag",
        primitives=("exact_arithmetic", "dependency_graph", "branch_merge"),
        distractors=0,
        style=style,
        probe=probe,
    )
    task.metadata["dependency_depth"] = budget
    return GeneratedCase(
        task,
        tuple(nodes),
        "Evaluate only declared integer dependencies and preserve exact signed values at branch merges.",
        "The target value remains provisional until every numeric parent dependency is resolved.",
        f"Final integer: {answer}.",
    )


def _iterative_policy_case(
    rng: random.Random, index: int, capacity: int, probe: bool
) -> GeneratedCase:
    live = _capacity_target(capacity, rng, probe=probe)
    budget = live - 4
    domain = rng.choice(PROBE_DOMAINS if probe else TRAIN_DOMAINS)
    style = _surface_style(capacity, rng, probe)
    modulus = rng.choice((97, 101, 127, 251, 509, 997))
    value = rng.randrange(modulus)
    initial = value
    rules = []
    nodes = []
    for i in range(budget):
        add = rng.randint(1, 13)
        mul = rng.randint(2, 5)
        threshold = rng.randrange(modulus)
        if value % 2 == 0:
            value = (value * mul + add) % modulus
            branch = "even"
        elif value > threshold:
            value = (value - add) % modulus
            branch = "odd-high"
        else:
            value = (value + mul + add) % modulus
            branch = "odd-low"
        rules.append({"add": add, "mul": mul, "threshold": threshold})
        parent = (f"r{i - 1}",) if i else ("premises",)
        nodes.append(
            ReasoningNode(
                f"r{i}", f"Step {i + 1}: {branch} branch gives state {value} mod {modulus}.", parent
            )
        )
    answer = str(value)
    rule_text = _render_records(
        style,
        (f"[{i}:m={r['mul']},a={r['add']},h={r['threshold']}]" for i, r in enumerate(rules)),
        (
            f"at step {i + 1}, use multiplier {r['mul']}, offset {r['add']}, and threshold {r['threshold']}"
            for i, r in enumerate(rules)
        ),
    )
    prompt = (
        f"A {domain} policy state starts x={initial}, modulus M={modulus}. For each record [i:m,a,h] in order: "
        "if x is even set x=(m*x+a) mod M; else if x>h set x=(x-a) mod M; otherwise set x=(x+m+a) mod M. "
        f"Records: {rule_text}. What is the final x? Answer with the integer only."
    )
    spec = {"kind": "iterative_policy", "initial": initial, "modulus": modulus, "rules": rules}
    task = _task(
        task_id=f"{'probe' if probe else 'comp'}-policy-c{capacity}-{index:06d}",
        family="iterative_policy",
        prompt=prompt,
        answer=answer,
        spec=spec,
        capacity=capacity,
        live_cells=live,
        domain=domain,
        topology="state_machine_chain",
        primitives=("branching_policy", "iterative_state", "modular_arithmetic"),
        distractors=0,
        style=style,
        probe=probe,
    )
    task.metadata["dependency_depth"] = budget
    return GeneratedCase(
        task,
        tuple(nodes),
        "Apply the ordered policy records to one evolving state; each branch uses the current state only.",
        "The final state is provisional until all ordered branch decisions are executed.",
        f"Final state: {answer}.",
    )


def _internal_repair_case(
    rng: random.Random, index: int, capacity: int, probe: bool
) -> GeneratedCase:
    live = _capacity_target(capacity, rng, probe=probe)
    budget = live - 4
    domain = rng.choice(PROBE_DOMAINS if probe else TRAIN_DOMAINS)
    style = _surface_style(capacity, rng, probe)
    labels = _labels("flag", budget + 1, rng, natural_limit=14)
    root = bool(rng.getrandbits(1))
    parity = [bool(rng.getrandbits(1)) for _ in range(budget)]
    values = [root]
    for flip in parity:
        values.append(values[-1] ^ flip)
    true_answer = values[-1]
    claimed = not true_answer
    nodes = []
    for i, value in enumerate(values[1:]):
        parent = (f"r{i - 1}",) if i else ("premises",)
        if i == max(1, budget // 3):
            text = f"The advisory shortcut conflicts here: authoritative propagation gives {labels[i + 1]}={'T' if value else 'F'}."
            roles = (CognitiveRole.CONSTRAINT, CognitiveRole.HYPOTHESIS)
        else:
            text = (
                f"Authoritative parity propagation gives {labels[i + 1]}={'T' if value else 'F'}."
            )
            roles = (CognitiveRole.PERCEPT, CognitiveRole.PLAN)
        nodes.append(ReasoningNode(f"r{i}", text, parent, roles))
    answer = "true" if true_answer else "false"
    constraints = _render_records(
        style,
        (
            f"{labels[i]}~{labels[i + 1]}:{'different' if parity[i] else 'same'}"
            for i in range(budget)
        ),
        (
            f"{labels[i]} and {labels[i + 1]} must be {'different' if parity[i] else 'the same'}"
            for i in range(budget)
        ),
    )
    prompt = (
        f"In a {domain} boolean chain, {labels[0]}={'true' if root else 'false'}. Authoritative constraints: {constraints}. "
        f"A non-authoritative shortcut note claims {labels[-1]}={'true' if claimed else 'false'} and may be wrong. "
        f"What does the authoritative chain imply for {labels[-1]}? Answer exactly `true` or `false`."
    )
    spec = {
        "kind": "internal_hypothesis_repair",
        "root": root,
        "parity": parity,
        "claimed": claimed,
    }
    task = _task(
        task_id=f"{'probe' if probe else 'comp'}-repair-c{capacity}-{index:06d}",
        family="internal_hypothesis_repair",
        prompt=prompt,
        answer=answer,
        spec=spec,
        capacity=capacity,
        live_cells=live,
        domain=domain,
        topology="conflict_then_repair",
        primitives=("provisional_hypothesis", "contradiction", "local_revision", "parity"),
        distractors=1,
        style=style,
        probe=probe,
        correction=True,
    )
    task.metadata["dependency_depth"] = budget
    initial_hypothesis = (
        f"The shortcut suggests {labels[-1]}={'T' if claimed else 'F'}, but it is only provisional."
    )
    return GeneratedCase(
        task,
        tuple(nodes),
        "Separate the advisory shortcut from authoritative parity constraints and test it against the full chain.",
        initial_hypothesis,
        f"Final authoritative value: {answer}.",
        correction=True,
    )


_GENERATORS = {
    "boolean_dag": _boolean_dag_case,
    "blocked_reachability": _blocked_reachability_case,
    "spatial_intervention": _spatial_intervention_case,
    "causal_intervention": _causal_intervention_case,
    "ordering_mesh": _ordering_mesh_case,
    "quantifier_dag": _quantifier_dag_case,
    "candidate_elimination": _candidate_elimination_case,
    "numeric_dag": _numeric_dag_case,
    "iterative_policy": _iterative_policy_case,
    "internal_hypothesis_repair": _internal_repair_case,
}


def _recompute(task: TeacherTask) -> str:
    spec = dict(task.metadata["logic_spec"])
    kind = str(spec["kind"])
    if kind == "boolean_dag":
        values = [bool(item) for item in spec["seeds"]]
        for rule in spec["rules"]:
            values.append(_bool_op(str(rule["op"]), values[int(rule["a"])], values[int(rule["b"])]))
        return "true" if values[-1] else "false"
    if kind == "blocked_reachability":
        adjacency: dict[int, list[int]] = defaultdict(list)
        for a, b in spec["edges"]:
            adjacency[int(a)].append(int(b))
        blocked = {int(item) for item in spec["blocked"]}
        queue = deque([int(spec["start"])])
        seen = {int(spec["start"])}
        while queue:
            cur = queue.popleft()
            for nxt in adjacency[cur]:
                if nxt not in blocked and nxt not in seen:
                    seen.add(nxt)
                    queue.append(nxt)
        return "reachable" if int(spec["target"]) in seen else "unreachable"
    if kind == "spatial_intervention":
        deltas = {"N": (0, 1), "S": (0, -1), "E": (1, 0), "W": (-1, 0)}
        steps = list(spec["steps"])
        steps[int(spec["changed"])] = str(spec["replacement"])
        dx = dy = 0
        for step in steps:
            sx, sy = deltas[str(step)]
            dx += sx
            dy += sy
        return _direction(dx, dy)
    if kind == "causal_intervention":
        values = [bool(item) for item in spec["seeds"]]
        intervention = int(spec["intervention"])
        for i, rule in enumerate(spec["rules"]):
            node = 3 + i
            if node == intervention:
                values.append(bool(spec["value"]))
            else:
                values.append(
                    _bool_op(str(rule["op"]), values[int(rule["a"])], values[int(rule["b"])])
                )
        return "true" if values[-1] else "false"
    if kind == "ordering_mesh":
        return str(spec["order"][int(spec["position"]) - 1])
    if kind == "quantifier_dag":
        adjacency: dict[int, list[int]] = defaultdict(list)
        for a, b in spec["inclusions"]:
            adjacency[int(a)].append(int(b))
        root, query = int(spec["root"]), int(spec["query"])
        queue = deque([root])
        closure = {root}
        while queue:
            cur = queue.popleft()
            for nxt in adjacency[cur]:
                if nxt not in closure:
                    closure.add(nxt)
                    queue.append(nxt)
        if query in closure:
            return "entailed"
        for a, b in spec["disjoint"]:
            a, b = int(a), int(b)
            if (a in closure and query == b) or (b in closure and query == a):
                return "contradicted"
        return "unknown"
    if kind == "candidate_elimination":
        signatures = list(spec["signatures"])
        features = [int(item) for item in spec["features"]]
        wanted = str(spec["wanted"])
        candidates = list(spec["candidates"])
        matches = [
            i
            for i, signature in enumerate(signatures)
            if all(signature[f] == wanted[f] for f in features)
        ]
        if len(matches) != 1:
            raise ValueError(f"candidate verifier expected unique match, found {matches}")
        return str(candidates[matches[0]])
    if kind == "numeric_dag":
        values = [int(item) for item in spec["seeds"]]
        for rule in spec["rules"]:
            a, b = values[int(rule["a"])], values[int(rule["b"])]
            op = str(rule["op"])
            value = {"add": a + b, "sub": a - b, "max": max(a, b), "min": min(a, b)}[op]
            values.append(max(-9999, min(9999, value)))
        return str(values[-1])
    if kind == "iterative_policy":
        value = int(spec["initial"])
        modulus = int(spec["modulus"])
        for rule in spec["rules"]:
            add, mul, threshold = int(rule["add"]), int(rule["mul"]), int(rule["threshold"])
            if value % 2 == 0:
                value = (mul * value + add) % modulus
            elif value > threshold:
                value = (value - add) % modulus
            else:
                value = (value + mul + add) % modulus
        return str(value)
    if kind == "internal_hypothesis_repair":
        value = bool(spec["root"])
        for flip in spec["parity"]:
            value ^= bool(flip)
        return "true" if value else "false"
    raise ValueError(f"unsupported compositional verifier kind: {kind}")


def _logic_spec_fingerprint(task: TeacherTask) -> str:
    return json.dumps(task.metadata["logic_spec"], sort_keys=True, separators=(",", ":"))


def _iter_case_groups(
    config: CompositionalTrainingConfig,
    *,
    probe: bool,
    forbidden_logic_specs: set[str] | None = None,
) -> Iterable[tuple[int, str, tuple[GeneratedCase, ...]]]:
    rng = random.Random(config.seed + (100_003 if probe else 0))
    capacity_counts = dict(config.probe_capacity_counts if probe else config.train_capacity_counts)
    seen_prompts: set[tuple[str, str]] = set()
    for capacity, total in sorted(capacity_counts.items()):
        per_family = total // len(COMPOSITIONAL_FAMILIES)
        for family_index, family in enumerate(COMPOSITIONAL_FAMILIES):
            generator = _GENERATORS[family]
            cases: list[GeneratedCase] = []
            for local_index in range(per_family):
                global_index = family_index * 100_000 + capacity * 10_000 + local_index
                for _attempt in range(1000):
                    case = generator(rng, global_index, capacity, probe)
                    semantic_key = (
                        " ".join(case.task.prompt.casefold().split()),
                        str(case.task.reference_answer).casefold(),
                    )
                    logic_fingerprint = _logic_spec_fingerprint(case.task)
                    if (
                        semantic_key not in seen_prompts
                        and (
                            forbidden_logic_specs is None
                            or logic_fingerprint not in forbidden_logic_specs
                        )
                    ):
                        seen_prompts.add(semantic_key)
                        break
                else:
                    raise RuntimeError(
                        f"could not generate a unique {family} case after 1000 attempts"
                    )
                expected = _recompute(case.task)
                if expected != str(case.task.reference_answer):
                    raise ValueError(
                        f"exact verifier mismatch for {case.task.task_id}: {expected} != {case.task.reference_answer}"
                    )
                cases.append(case)
            yield capacity, family, tuple(cases)


def _trajectory_transition_count(trajectories: Iterable[TrajectoryExample]) -> int:
    total = 0
    for example in trajectories:
        steps = [target.step for target in example.thought_targets]
        steps.extend(target.step for target in example.display_targets)
        total += max(steps, default=0)
    return total


def _audit(
    tasks: tuple[TeacherTask, ...],
    plans: tuple[TeacherPlan, ...],
    trajectories: tuple[TrajectoryExample, ...],
) -> dict[str, Any]:
    plan_by_id = {plan.task_id: plan for plan in plans}
    family_counts: Counter[str] = Counter()
    capacity_counts: Counter[int] = Counter()
    style_counts: Counter[str] = Counter()
    domain_counts: Counter[str] = Counter()
    depth_buckets: Counter[str] = Counter()
    max_text = 0
    max_live_cells = 0
    anchor_instances = 0
    link_instances = 0
    exact_failures: list[str] = []
    for task in tasks:
        family_counts[str(task.metadata["family"])] += 1
        capacity_counts[int(task.metadata["thought_capacity_bucket"])] += 1
        style_counts[str(task.metadata["surface_style"])] += 1
        domain_counts[str(task.metadata["domain"])] += 1
        depth = int(task.metadata.get("dependency_depth", 0))
        depth_buckets[
            "1-8"
            if depth <= 8
            else "9-16"
            if depth <= 16
            else "17-32"
            if depth <= 32
            else "33-64"
            if depth <= 64
            else "65+"
        ] += 1
        if _recompute(task) != str(task.reference_answer):
            exact_failures.append(task.task_id)
        plan = plan_by_id[task.task_id]
        for frame in plan.frames:
            max_live_cells = max(
                max_live_cells,
                sum(cell.lifecycle is not CellLifecycle.RETIRED for cell in frame.cells),
            )
            for item in frame.cells:
                max_text = max(max_text, len(item.semantic_text))
                anchor_instances += len(item.anchors)
                link_instances += len(item.links)
                if len(item.semantic_text) > SEMANTIC_TEXT_CAP:
                    raise ValueError(f"semantic text cap exceeded in {task.task_id}/{item.cell_id}")
    if exact_failures:
        raise ValueError(f"exact verifier failures: {exact_failures[:10]}")
    trajectory_capacity_max = max(
        (target.slot + 1 for example in trajectories for target in example.thought_targets),
        default=0,
    )
    return {
        "family_counts": dict(sorted(family_counts.items())),
        "capacity_bucket_counts": {str(k): v for k, v in sorted(capacity_counts.items())},
        "surface_style_counts": dict(sorted(style_counts.items())),
        "domain_count": len(domain_counts),
        "dependency_depth_buckets": dict(sorted(depth_buckets.items())),
        "max_semantic_text_chars": max_text,
        "max_live_cells": max_live_cells,
        "max_physical_slot_index_plus_one": trajectory_capacity_max,
        "anchor_instances": anchor_instances,
        "link_instances": link_instances,
        "exact_verifier_failures": 0,
        "compiled_transitions": _trajectory_transition_count(trajectories),
    }


def _write_jsonl_block(handle: Any, records: Iterable[Any]) -> None:
    for record in records:
        if isinstance(record, TrajectoryExample):
            payload = trajectory_to_dict(record)
        else:
            payload = record.to_dict() if hasattr(record, "to_dict") else record
        handle.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")


def build_compositional_training_streaming(
    output_dir: str | Path,
    config: CompositionalTrainingConfig | None = None,
) -> dict[str, Any]:
    """Build the production corpus without retaining cross-family plans in memory."""

    config = config or CompositionalTrainingConfig()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    train_paths = {
        "tasks": output / "compositional-teacher-tasks-v1.jsonl",
        "plans": output / "compositional-teacher-plans-v1.jsonl",
        "accepted": output / "compositional-teacher-plans-v1.accepted.jsonl",
        "reviews": output / "compositional-teacher-review-v1.jsonl",
        "trajectories": output / "compositional-trajectories-v1.jsonl",
        "trajectory_manifest": output / "compositional-trajectories-v1.manifest.json",
    }
    probe_paths = {
        "tasks": output / "generalization-probe-tasks-v1.jsonl",
        "plans": output / "generalization-probe-plans-v1.jsonl",
        "reviews": output / "generalization-probe-review-v1.jsonl",
        "trajectories": output / "generalization-probe-trajectories-v1.jsonl",
        "manifest": output / "generalization-probe-v1.manifest.json",
    }

    plan_rng = random.Random(config.seed + 77)

    def stream_split(
        *,
        probe: bool,
        forbidden_logic_specs: set[str] | None = None,
    ) -> tuple[dict[str, Any], int, int, set[str]]:
        paths = probe_paths if probe else train_paths
        variants = config.probe_variants_per_task if probe else config.variants_per_task
        family_counts: Counter[str] = Counter()
        capacity_counts: Counter[int] = Counter()
        style_counts: Counter[str] = Counter()
        depth_buckets: Counter[str] = Counter()
        domains: set[str] = set()
        max_text = 0
        max_live = 0
        max_slot = 0
        anchors = 0
        links = 0
        transitions = 0
        task_count = 0
        trajectory_count = 0
        logic_specs: set[str] = set()

        handles = {
            name: path.open("w", encoding="utf-8")
            for name, path in paths.items()
            if name not in {"manifest", "trajectory_manifest"}
        }
        try:
            for capacity, family, cases in _iter_case_groups(
                config,
                probe=probe,
                forbidden_logic_specs=forbidden_logic_specs,
            ):
                tasks = tuple(case.task for case in cases)
                logic_specs.update(_logic_spec_fingerprint(task) for task in tasks)
                plans = tuple(_plan_for(case, plan_rng) for case in cases)
                reviews = review_teacher_plans(tasks, plans)
                rejected = tuple(review for review in reviews if not review.accepted)
                if rejected:
                    raise ValueError(
                        f"compositional teacher review rejected {len(rejected)} plans; "
                        f"first={rejected[0].to_dict()}"
                    )
                family_index = COMPOSITIONAL_FAMILIES.index(family)
                trajectories = compile_teacher_plans(
                    tasks,
                    plans,
                    TeacherScheduleConfig(
                        thought_capacity=capacity,
                        min_delay_steps=1,
                        max_delay_steps=4,
                        variants_per_task=variants,
                        seed=(
                            config.seed
                            + (200_003 if probe else 0)
                            + capacity * 257
                            + family_index * 1009
                        ),
                    ),
                )
                report = _audit(tasks, plans, trajectories)

                _write_jsonl_block(handles["tasks"], tasks)
                _write_jsonl_block(handles["plans"], plans)
                if not probe:
                    _write_jsonl_block(handles["accepted"], plans)
                _write_jsonl_block(handles["reviews"], reviews)
                _write_jsonl_block(handles["trajectories"], trajectories)

                family_counts.update(report["family_counts"])
                capacity_counts.update(
                    {int(key): value for key, value in report["capacity_bucket_counts"].items()}
                )
                style_counts.update(report["surface_style_counts"])
                depth_buckets.update(report["dependency_depth_buckets"])
                domains.update(str(task.metadata["domain"]) for task in tasks)
                max_text = max(max_text, int(report["max_semantic_text_chars"]))
                max_live = max(max_live, int(report["max_live_cells"]))
                max_slot = max(max_slot, int(report["max_physical_slot_index_plus_one"]))
                anchors += int(report["anchor_instances"])
                links += int(report["link_instances"])
                transitions += int(report["compiled_transitions"])
                task_count += len(tasks)
                trajectory_count += len(trajectories)
        finally:
            for handle in handles.values():
                handle.close()

        audit = {
            "family_counts": dict(sorted(family_counts.items())),
            "capacity_bucket_counts": {str(k): v for k, v in sorted(capacity_counts.items())},
            "surface_style_counts": dict(sorted(style_counts.items())),
            "domain_count": len(domains),
            "dependency_depth_buckets": dict(sorted(depth_buckets.items())),
            "max_semantic_text_chars": max_text,
            "max_live_cells": max_live,
            "max_physical_slot_index_plus_one": max_slot,
            "anchor_instances": anchors,
            "link_instances": links,
            "exact_verifier_failures": 0,
            "compiled_transitions": transitions,
        }
        return audit, task_count, trajectory_count, logic_specs

    train_audit, train_task_count, train_trajectory_count, train_logic_specs = stream_split(
        probe=False
    )
    probe_audit, probe_task_count, probe_trajectory_count, probe_logic_specs = stream_split(
        probe=True,
        forbidden_logic_specs=train_logic_specs,
    )
    exact_logic_spec_overlap = len(train_logic_specs & probe_logic_specs)
    if exact_logic_spec_overlap:
        raise AssertionError("generalization probe contains training logic specs")

    train_manifest = {
        "format_version": 1,
        "name": "compositional-longtail-reasoning-v1",
        "version": 1,
        "generator": "GPT-5.6-Sol-authored deterministic compositional curriculum",
        "seed": config.seed,
        "semantic_tasks": train_task_count,
        "accepted_plans": train_task_count,
        "review_rejected": 0,
        "compiled_trajectories": train_trajectory_count,
        "compiled_transitions": train_audit["compiled_transitions"],
        "thought_capacity_curriculum": [8, 16, 32, 64, 128],
        "thought_capacity_required": 128,
        "semantic_text_cap": SEMANTIC_TEXT_CAP,
        "ood_probe_excluded_from_training": True,
        "audit": train_audit,
        "tasks_sha256": _sha256(train_paths["tasks"]),
        "plans_sha256": _sha256(train_paths["plans"]),
        "accepted_plans_sha256": _sha256(train_paths["accepted"]),
        "review_sha256": _sha256(train_paths["reviews"]),
        "compiled_sha256": _sha256(train_paths["trajectories"]),
    }
    probe_manifest = {
        "format_version": 1,
        "name": "compositional-generalization-probe-v1",
        "version": 1,
        "seed": config.seed + 100_003,
        "semantic_tasks": probe_task_count,
        "compiled_trajectories": probe_trajectory_count,
        "compiled_transitions": probe_audit["compiled_transitions"],
        "thought_capacity_required": 128,
        "training_eligible": False,
        "strict_holdout_axes": ["domain", "exact_logic_spec"],
        "exact_logic_spec_overlap_with_training": exact_logic_spec_overlap,
        "generalization_axes": [
            "unseen_domain",
            "higher_dependency_depth",
            "capacity_tail",
            "long_range_dependency_density",
            "surface_rephrasing",
            "graph_topology_shift",
        ],
        "audit": probe_audit,
        "tasks_sha256": _sha256(probe_paths["tasks"]),
        "plans_sha256": _sha256(probe_paths["plans"]),
        "review_sha256": _sha256(probe_paths["reviews"]),
        "compiled_sha256": _sha256(probe_paths["trajectories"]),
    }
    train_trajectory_manifest = {
        "format_version": 1,
        "name": "compositional-trajectories-v1",
        "schema": "cid.TrajectoryExample.v1",
        "examples": train_trajectory_count,
        "transitions": train_audit["compiled_transitions"],
        "thought_capacity_required": 128,
        "capacity_bucket_counts": {
            key: value * config.variants_per_task
            for key, value in train_audit["capacity_bucket_counts"].items()
        },
        "sha256": train_manifest["compiled_sha256"],
    }
    train_paths["trajectory_manifest"].write_text(
        json.dumps(train_trajectory_manifest, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    probe_paths["manifest"].write_text(
        json.dumps(probe_manifest, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    (output.parent / "compositional-teacher-v1.reference-manifest.json").write_text(
        json.dumps(train_manifest, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "train_manifest": train_manifest,
        "probe_manifest": probe_manifest,
        "train_tasks": train_task_count,
        "train_trajectories": train_trajectory_count,
        "probe_tasks": probe_task_count,
        "probe_trajectories": probe_trajectory_count,
    }
