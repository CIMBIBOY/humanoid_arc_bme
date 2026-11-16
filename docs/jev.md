# Jev adapter (optional, semantic tier only)

Jev is a cloud decision model (documented latency 70-500 ms). The adapter `onboard/command/jev.py` is an
`IntentClassifier` for `CompositeParser`. It is asked only about sentences the grammar rejected, and only for an
intent out of a closed list. Numbers, objects and colours still come from the grammar. Jev is never in the stop
path and the robot never depends on it.

    sentence -> LocalGrammarParser (STOP, all valid commands) -> understood: done, Jev not called
                                                               -> rejected: JevClassifier -> intent -> parse_as

## Placeholder schema (NOT the real API; every constant is tagged PLACEHOLDER in the code)

- request: `{"input": text, "choices": [options], "context": {"visible": [...]}}`, POST, JSON
- response: `{"choice": str, "probabilities": {option: p}}` or `{"choice": str, "probability": p}`
- auth: `Authorization: Bearer <key>`, key read from `$JEV_API_KEY` at call time, never logged
- budget: one attempt, `timeout_s=0.8`, no retries; breaker opens after 3 consecutive failures for 30 s

## Failure behaviour

| Situation | Result |
|---|---|
| timeout, transport error | exception, grammar's own Reject, robot does nothing |
| outage (breaker open) | `JevUnavailable` at once, no network call, until cooldown ends, then one trial call |
| malformed answer | `ValueError` inside the adapter -> `JevUnavailable` -> grammar Reject |
| probability < `min_prob` (0.8) | grammar Reject |
| missing API key | `JevUnavailable`, no network call |
| stop sentence | grammar returns Stop first; Jev is not called |

## Offline use

`FakeJev` (modes ok, timeout, error, malformed, low_prob, outage_after:N), `ReplayJev` (recorded
`{text: [choice, probability]}` file, unrecorded sentence -> KeyError -> grammar) and `record(...)` mean tests and
simulator runs never touch the network. Run log fields: `last_latency_s`, `calls`, `failures`, `state`.

## What the user must supply

1. The real API specification: URL, auth header and scheme, request and response field names.
2. Whether probabilities are returned (if not, the confidence gate cannot work as written).
3. The API key, in `$JEV_API_KEY`, and the rate limits (the 3-failure breaker may need tuning).
4. Whether Jev's model/version selection needs a field (`model` is currently a guessed header).
