---
name: evaluate-realism
description: Independently evaluate a frozen realism-stage candidate after trajectory reconstruction, separating natural appearance, reference fidelity, camera processing and physical nonregression. Use for stage realism or a refreshed baseline before synthesis.
---

# Evaluate a realism candidate

Read the request, observation restrictions, `rubrics/realism.yaml` and the
[executable evidence contract](../../docs/realism_stage.md). Write only in the evaluation
folder; do not alter the candidate, protocol, masks or acceptance thresholds.

1. Recompute the diagnostics against the candidate's authorized payload with
   `any2sim.realism.check(candidate_path, out=absolute_evaluation_check_path)` in the base
   environment. This read-only candidate check writes only the specified evaluation JSON;
   do not run a receipt-writing tool directly on the frozen candidate. For additional tool
   runs, copy the candidate into an isolated workspace inside the evaluation folder. Verify
   its baseline matches the frozen candidate. A builder's ready status is not independent acceptance.
2. Inspect original references, same-pose signal-chain crops, linear light passes and
   native near/far surface regions. Verify declared masks exclude moving objects, changing
   shadows, clipped areas and duplicate/near-duplicate content; indices alone do not prove
   held-out support. Check actual compiled materials/UVs where their effect is uncertain.
3. Judge every rubric item separately, including support/grasp/nesting boundaries and
   thin surfaces when present. Use the causal diagnostics in
   [render-realism](../render-realism/SKILL.md). A physically plausible green bounce may
   still be too strong, but do not blame it without separated evidence.
4. Inspect the continuous nominal sequence, a consecutive-frame overview and native
   windows around detected peak changes plus grasp/release. State duration/views/coverage.
   Duplicate or high-change counts are descriptive; inspect the underlying motion.
5. Report `perceptual_realism` separately from `reference_fidelity`. At usable fidelity,
   a missing paired camera reference may have an explicitly limited scope; do not claim
   calibration or likeness for it. At precise fidelity, missing paired views block pass.
   Unknown findings on required criteria block the overall verdict.
6. Write `report.yaml` and `realism_review.json` under `evaluations/<k>/`. The JSON follows
   schema v2 in the evidence contract, with distinct builder/evaluator identities, the
   exact candidate fingerprint, notes, evidence paths and hashes **computed now**. Include
   the independent diagnostics and all their input hashes. This is an auditable identity
   declaration, not cryptographic proof of reviewer independence.

The YAML report names `stage: realism`, `candidate: candidates/<k>` and `verdict`;
each rubric criterion has `status: pass|fail|unknown`, metrics, evidence and a note.
List unresolved issues with an owner and concrete next action. Unknown blocks pass.

The builder may publish the unchanged decision with `realism_check action=adopt` only
when the current assets and evidence still match. The publisher does not convert
`revise`, `unknown`, missing evidence or expired hashes into a pass. A diagnostic render
exception never completes the realism stage.
