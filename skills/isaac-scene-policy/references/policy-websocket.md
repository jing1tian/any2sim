# WebSocket policy contract

Inspect the server implementation/configuration before selecting transport and observation fields. OpenPI's common protocol sends one binary msgpack metadata greeting, then accepts observation dictionaries and returns dictionaries with an `actions` array. NumPy arrays use OpenPI's byte-keyed ndarray marker/dtype/shape/data encoding, not arbitrary JSON or generic msgpack-numpy pickle fallback. A text reply may be a server exception. Avoid unbounded reconnect loops when evaluating.

Document and validate:

- State joint order, dimensions, radians versus normalized values, absolute targets versus deltas/Cartesian commands. A server trained on a different embodiment needs an explicit adapter; never silently truncate or pad actions.
- Observation keys, prompt, camera names, uint8 RGB HWC layout and resolution. Any preprocessing must match training. Do not assume that all OpenPI checkpoints use `state`/`image` or the same robot joints.
- Action chunk shape `[T,D]`, action period and how many returned steps execute before replanning. Clear queued actions at every episode reset and after failures. With absolute targets, each row holds for one control interval; temporal interpolation is a separate declared choice.
- Connection/receive timeouts, finite action validation and payload bounds. Do not execute stale or fallback actions after a transport failure. Save partial rollout evidence and a failed status, without counting the run as a completed benchmark.
- Simulated time versus wall time. Blocking inference should pause advancement of simulation; report inference latency separately from task simulated duration. Rendering observations also must not advance physics.

OpenPI's standard serving loop has no episode-reset RPC. Opening a new connection clears the client's queue but does **not** guarantee reset of a server-global recurrent policy. Require a server-specific reset extension for stateful models; do not invent an unsupported reset payload. If metadata advertises joint/action semantics, compare them to the configured contract. Otherwise the user/configuration must explicitly establish the contract.

A small local mock server tests transport, chunk handling, RGB serialization, timeouts, invalid responses and reset isolation. Independently test compatibility against the server's codec. Then run a short native Isaac closed loop before connecting a trained checkpoint. Log inference timing, model identity when supplied, controller mode, seeds and requested/applied actions. Separate transport correctness from task success and cross-engine physics fidelity.
