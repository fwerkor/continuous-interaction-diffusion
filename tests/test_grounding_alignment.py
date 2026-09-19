from __future__ import annotations

from types import SimpleNamespace

import torch

from cid.model import losses


class _ReferenceAssignmentEngine:
    @staticmethod
    def batched_linear_assignment(
        costs: torch.Tensor,
        row_counts: torch.Tensor,
    ) -> torch.Tensor:
        rows = costs.shape[1]
        matrices = costs.detach().float().cpu().tolist()
        counts = row_counts.detach().long().cpu().tolist()
        assignments = [
            list(losses._linear_assignment_matrix(matrix[:count]))
            + [-1] * (rows - count)
            for matrix, count in zip(matrices, counts, strict=True)
        ]
        return torch.tensor(
            assignments,
            dtype=torch.long,
            device=costs.device,
        )


def _alignment_case(seed: int) -> tuple[SimpleNamespace, SimpleNamespace]:
    generator = torch.Generator().manual_seed(seed)
    batch = 2
    cells = 7
    hidden = 11
    anchor_slots = 4
    link_slots = 8
    anchor_kinds = 4
    link_relations = 6
    object_kinds = 5

    anchor_presence_mask = torch.zeros(batch, cells, anchor_slots, dtype=torch.bool)
    link_presence_mask = torch.zeros(batch, cells, link_slots, dtype=torch.bool)
    anchor_mask = torch.zeros_like(anchor_presence_mask)
    link_mask = torch.zeros_like(link_presence_mask)

    anchor_kind_targets = torch.full(
        (batch, cells, anchor_slots),
        -100,
        dtype=torch.long,
    )
    link_relation_targets = torch.full(
        (batch, cells, link_slots),
        -100,
        dtype=torch.long,
    )
    link_target_kind_targets = torch.full(
        (batch, cells, link_slots),
        -100,
        dtype=torch.long,
    )
    anchor_embeddings = torch.zeros(batch, cells, anchor_slots, hidden)
    link_target_embeddings = torch.zeros(batch, cells, link_slots, hidden)

    for batch_index in range(batch):
        for cell in range(cells):
            supervised = (batch_index + cell + seed) % 4 != 0
            if not supervised:
                continue
            anchor_presence_mask[batch_index, cell] = True
            link_presence_mask[batch_index, cell] = True

            anchor_count = (batch_index + cell + seed) % (anchor_slots + 1)
            if anchor_count:
                anchor_indices = torch.randperm(
                    anchor_slots,
                    generator=generator,
                )[:anchor_count].sort().values
                anchor_mask[batch_index, cell, anchor_indices] = True
                anchor_kind_targets[batch_index, cell, anchor_indices] = torch.randint(
                    anchor_kinds,
                    (anchor_count,),
                    generator=generator,
                )
                anchor_embeddings[batch_index, cell, anchor_indices] = torch.randn(
                    anchor_count,
                    hidden,
                    generator=generator,
                )

            link_count = (2 * batch_index + cell + seed) % (link_slots + 1)
            if link_count:
                link_indices = torch.randperm(
                    link_slots,
                    generator=generator,
                )[:link_count].sort().values
                link_mask[batch_index, cell, link_indices] = True
                link_relation_targets[batch_index, cell, link_indices] = torch.randint(
                    link_relations,
                    (link_count,),
                    generator=generator,
                )
                link_target_kind_targets[batch_index, cell, link_indices] = torch.randint(
                    object_kinds,
                    (link_count,),
                    generator=generator,
                )
                link_target_embeddings[batch_index, cell, link_indices] = torch.randn(
                    link_count,
                    hidden,
                    generator=generator,
                )

    output = SimpleNamespace(
        anchor_query=torch.randn(
            batch,
            cells,
            anchor_slots,
            hidden,
            generator=generator,
        ),
        anchor_kind_logits=torch.randn(
            batch,
            cells,
            anchor_slots,
            anchor_kinds,
            generator=generator,
        ),
        link_target_query=torch.randn(
            batch,
            cells,
            link_slots,
            hidden,
            generator=generator,
        ),
        link_relation_logits=torch.randn(
            batch,
            cells,
            link_slots,
            link_relations,
            generator=generator,
        ),
        link_target_kind_logits=torch.randn(
            batch,
            cells,
            link_slots,
            object_kinds,
            generator=generator,
        ),
    )
    targets = SimpleNamespace(
        anchor_presence_targets=torch.zeros(
            batch,
            cells,
            anchor_slots,
        ),
        anchor_presence_mask=anchor_presence_mask,
        anchor_kind_targets=anchor_kind_targets,
        anchor_embeddings=anchor_embeddings,
        anchor_mask=anchor_mask,
        link_presence_targets=torch.zeros(
            batch,
            cells,
            link_slots,
        ),
        link_presence_mask=link_presence_mask,
        link_relation_targets=link_relation_targets,
        link_target_kind_targets=link_target_kind_targets,
        link_target_embeddings=link_target_embeddings,
        link_mask=link_mask,
    )
    return output, targets


def test_batched_grounding_alignment_matches_fallback(
    monkeypatch,
) -> None:
    for seed in range(12):
        output, targets = _alignment_case(seed)

        monkeypatch.setattr(losses, "cuda_engine", lambda *args, **kwargs: None)
        expected_anchor = losses._align_anchor_targets(output, targets)
        expected_link = losses._align_link_targets(output, targets)

        monkeypatch.setattr(
            losses,
            "cuda_engine",
            lambda *args, **kwargs: _ReferenceAssignmentEngine(),
        )
        actual_anchor = losses._align_anchor_targets(output, targets)
        actual_link = losses._align_link_targets(output, targets)

        for actual, expected in zip(actual_anchor, expected_anchor, strict=True):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for actual, expected in zip(actual_link, expected_link, strict=True):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
