# Geminoria Security To-Do

- Add configurable strict redaction presets (registry-backed regex profiles).
- Add optional `neverSendNicks` mode for history-derived tool payloads.

## Geminoria Improvement Plan

- Add optional network-scoped overrides for `model`, `maxResults`, `maxToolRounds`, and `bufferSize`.
- Add channel-scoped persona/prompt profile overrides.
- Add locale auto-selection per network/channel with a global fallback, including `en-GB`, `en-AU`, and `en-US`.
- Expand `@gemcache` with per-network stats and per-network clear.
- Add configurable Gemini timeout/retry settings.
- Add response formatting modes: single-line, wrapped, and paged.
- Expand `@gemdiag` to display the active network/channel policy snapshot.
- Add regression tests for each new config precedence rule before implementing it.
