from __future__ import annotations

import argparse
import asyncio
import time

from cid.contracts import FreshnessDemand, InformationNeed, ModelContext, ModelUpdate
from cid.grounding import ObjectRef
from cid.metrics import summarize_runtime
from cid.runtime import CIDRuntime, RuntimeConfig, SourceRegistry, StaticMappingSource
from cid.state import CognitiveField, CognitiveRole, DisplayCanvas


class DemoPolicy:
    def __init__(self, target_cell_id: str) -> None:
        self.target_cell_id = target_cell_id

    def step(self, context: ModelContext) -> ModelUpdate:
        time.sleep(0.02)
        need = InformationNeed(
            need_id="latency-spec",
            source_scores={"docs": 1.0},
            arguments={"key": "latency_ms"},
            confidence=1.0,
            freshness=FreshnessDemand.ONCE,
            target_cells=(ObjectRef.cell(self.target_cell_id),),
            target_display=(ObjectRef.display_span(0, 1),),
            promote_to_fact=True,
        )
        if context.percepts:
            value = int(context.percepts[0].observation.value)
            display = context.display.advance((value, 0, 0, 0))
            return ModelUpdate(
                thought=context.thought.advance(context.thought.cells),
                display=display,
                needs=(need,),
                converged=True,
            )
        return ModelUpdate(
            thought=context.thought.advance(context.thought.cells),
            display=context.display.advance(context.display.token_ids),
            needs=(need,),
        )


async def _run_demo() -> None:
    sources = SourceRegistry()
    sources.register(StaticMappingSource("docs", {"latency_ms": 37}, delay_s=0.03))
    runtime = CIDRuntime(sources, RuntimeConfig(max_steps=12, idle_yield_s=0.001))
    thought, need_cell_id = CognitiveField.empty(capacity=4, width=8).allocate(
        roles={CognitiveRole.INFORMATION_NEED: 1.0}
    )
    result = await runtime.run(
        DemoPolicy(need_cell_id),
        thought=thought,
        display=DisplayCanvas.masked(length=4, mask_token_id=-1),
    )
    metrics = summarize_runtime(result)
    print(f"converged={result.converged} steps={result.steps}")
    print(f"display={result.display.token_ids}")
    print(f"protected_facts={[(item.key, item.value) for item in result.facts]}")
    print(
        "interaction="
        f"external:{metrics.external_refreshes} "
        f"projections:{metrics.cognitive_projections} "
        f"model_steps_during_io:{metrics.model_steps_during_io}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(prog="cid")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("demo", help="run the asynchronous static-source demo")
    args = parser.parse_args()
    if args.command == "demo":
        asyncio.run(_run_demo())


if __name__ == "__main__":
    main()
