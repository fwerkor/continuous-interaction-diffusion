"""Controlled training test of contradictory multi-Need labels.

Uses real CID tensorizer and loss/assignment, with fixed actual CID backbone
features and only the two Need logits trainable. Not full-model quality evidence.
"""

from dataclasses import replace

import torch
from test_training import TinyTokenizer, _indistinguishable_multi_need_pair, make_adapter

from cid.model.losses import CIDLossWeights, cid_loss
from cid.model.training import ILLaDATrajectoryTensorizer


def run(enabled: bool) -> tuple[list[float], list[float]]:
    model = make_adapter(seed=191)
    tensorizer = ILLaDATrajectoryTensorizer(model, TinyTokenizer())
    e1, e2 = _indistinguishable_multi_need_pair()
    samples = tuple(
        tensorizer.tensorize(
            e, source_step=0, timestep=1.0, generator=torch.Generator().manual_seed(37)
        )
        for e in (e1, e2)
    )
    with torch.no_grad():
        out = model(samples[0].batch)
    logits = torch.nn.Parameter(torch.tensor([0.07, -0.07], dtype=torch.float32))
    optimizer = torch.optim.AdamW([logits], lr=0.1, weight_decay=0.0)
    weights = replace(
        CIDLossWeights(),
        thought=0,
        convergence=0,
        allocation=0,
        display=0,
        roles=0,
        uncertainty=0,
        noise=0,
        lifecycle=0,
        source=0,
        need_cell_route=0,
        need_display_route=0,
        argument_presence=0,
        argument_ground=0,
        revision=0,
        refresh=0,
        anchor_presence=0,
        anchor_kind=0,
        anchor_ground=0,
        link_presence=0,
        link_relation=0,
        link_target_kind=0,
        link_ground=0,
    )
    losses = []
    for step in range(150):
        optimizer.zero_grad()
        expanded = out.need_logits.detach().clone()
        # The two candidate actions share the same model-visible state.
        expanded[0, 1, :2] = logits
        total = sum(
            cid_loss(
                replace(out, need_logits=expanded),
                sample.targets if enabled else replace(sample.targets, need_permutation_groups=()),
                weights=weights,
            ).total
            for sample in samples
        ) / len(samples)
        total.backward()
        optimizer.step()
        if step in (0, 49, 149):
            losses.append(float(total.detach()))
    return logits.detach().sigmoid().tolist(), losses


def main() -> None:
    torch.set_num_threads(2)
    old, old_loss = run(False)
    new, new_loss = run(True)
    print(
        {
            "old_probabilities": old,
            "old_losses": old_loss,
            "joint_probabilities": new,
            "joint_losses": new_loss,
            "old_threshold_0_6_selected": sum(v >= 0.6 for v in old),
            "joint_threshold_0_6_selected": sum(v >= 0.6 for v in new),
        }
    )


if __name__ == "__main__":
    main()
