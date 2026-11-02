# Diagnosing MuJoCo / PhysX differences

First distinguish kinematic playback from actuator-driven replay. Loading identical commanded positions into two engines does not imply identical achieved contact trajectories. Physics timestep, action interpolation, drive formulation, collision geometry, contact offsets, friction combination and solver behavior can change a marginal grasp.

Use controlled pairs rather than changing several parameters until a task passes:

1. Freeze source commands, reset state and variant. Audit joint-name mapping, frames and units. Compare pre-contact FK; a discrepancy before contact points toward mapping, initial state or free-space control.
2. Re-run a historically successful native episode in the new adapter. Compare overlapping joint/object histories numerically. Reproducing it helps distinguish a global import regression from trajectory-specific transfer failure.
3. Compare contact onset, peak lift, last finger contact, final containment/release and loaded finger tracking. Track actual contacting link names, not only object motion. Preserve sampling/aggregation semantics: max impulse/dt over a control interval differs from instantaneous force at its boundary.
4. Test initial-pose mismatch independently. Matching sub-millimeter reset differences may not resolve a contact/control failure.
5. Inspect material bindings on active partitioned colliders. Missing torsional patch settings on new shapes can survive an apparently correct original-mesh export. A radius derived from source torsional/sliding friction is only an approximation, not proof of solver equivalence.
6. Compare implicit PhysX drives against explicit source PD as a diagnostic. Avoid double-counting stiffness/damping or gravity/passive compensation. Preserve source gains, effort clamping order and physics timestep. Disable the selected implicit drives when applying corresponding explicit PD effort. Split arm-only and finger-only interventions to distinguish grasp control from arm tracking changes.
7. Test contact formulation changes in the source engine to assess sensitivity, while avoiding claims that a particular MuJoCo cone option is equivalent to PhysX. Solver and contact models differ beyond a single flag.

In one pepper grasp case, 6/6 sampled trajectories passed MuJoCo and 4/6 passed default PhysX. Explicit finger PD alone rescued the reference; arm-only PD did not. Full explicit PD rescued one generated failure but not another. This is evidence for a controller/contact interaction in that case, **not a universal setting** to apply to future scenes.

Explicit commanded torque is not measured actuator torque. When only one joint group uses explicit PD, the effort vector omits unselected implicit drive output. Label saturation denominators and measurement scope. Keep the original default until validation supports adopting a candidate for the intended episode/policy population.
