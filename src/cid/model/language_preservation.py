from __future__ import annotations

import torch
from torch import Tensor


def native_language_kl(
    student_logits: Tensor,
    native_logits: Tensor,
    eligible: Tensor,
    *,
    top_k: int = 16,
    chunk_size: int = 8,
) -> Tensor:
    """Preserve native language probabilities, independent of answer position.

    Coarsen the teacher distribution into its top-k tokens and one residual
    category. This avoids storing a full-vocabulary FP32 teacher distribution
    for every Display position. Gradients update only the CID logits.
    """
    if student_logits.shape != native_logits.shape or student_logits.ndim != 3:
        raise ValueError("native language logits must have matching [B,L,V] shapes")
    if eligible.shape != student_logits.shape[:2]:
        raise ValueError("native language preservation mask must match Display")
    if top_k < 1 or chunk_size < 1:
        raise ValueError("top_k and chunk_size must be positive")
    mask = eligible.bool()
    if not bool(mask.any()):
        return student_logits.sum() * 0.0
    total = student_logits.new_zeros((), dtype=torch.float32)
    for start in range(0, student_logits.shape[1], chunk_size):
        selected = mask[:, start:start + chunk_size]
        if not bool(selected.any()):
            continue
        student = student_logits[:, start:start + chunk_size].float()
        teacher = native_logits[:, start:start + chunk_size].detach().float()
        with torch.no_grad():
            teacher_log_norm = torch.logsumexp(teacher, dim=-1, keepdim=True)
            teacher_top, indices = teacher.topk(min(top_k, teacher.shape[-1] - 1), dim=-1)
            log_teacher = teacher_top - teacher_log_norm
            teacher_prob = log_teacher.exp()
            teacher_other = (1.0 - teacher_prob.sum(dim=-1)).clamp_min(1e-8)
        student_log_norm = torch.logsumexp(student, dim=-1, keepdim=True)
        log_student = student.gather(-1, indices) - student_log_norm
        student_other = (1.0 - log_student.exp().sum(dim=-1)).clamp_min(1e-8)
        divergence = (teacher_prob * (log_teacher - log_student)).sum(dim=-1)
        divergence = divergence + teacher_other * (
            teacher_other.log() - student_other.log()
        )
        total = total + (divergence * selected).sum()
    return total / mask.sum().clamp_min(1)
