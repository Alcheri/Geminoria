# Geminoria Architecture (Phase 2)

## Layout

```text
Geminoria/
├── plugin.py                    # Limnoria entrypoint (thin facade)
├── __init__.py                  # Plugin bootstrap + Limnoria integration
│
├── core/
│   ├── core.py                  # Main orchestration and tool loop
│   ├── system.py                # System prompt + Gemini tool declarations
│   ├── services.py              # Gemini service adapter (async thread loop)
│   ├── textutils.py             # Sanitizing, redaction, progress helpers
│   └── __init__.py
│
├── state/
│   ├── memory.py                # In-memory buffers + request slot controls
│   ├── cache.py                 # SQLite cache + similarity helpers
│   └── __init__.py
│
├── config/
│   ├── config.py                # Limnoria registry declarations
│   ├── config_runtime.py        # Runtime config dataclass + loader
│   └── __init__.py
│
└── tests/
    ├── test.py                  # Plugin smoke + utility behavior checks
    ├── test_architecture.py     # Architecture and import contract checks
    └── __init__.py
```

## Dependency Flow

```mermaid
flowchart TD
    P["plugin.py"] --> C["core/core.py"]
    P --> R["config/config_runtime.py"]
    P --> S["core/services.py"]
    P --> T["core/textutils.py"]
    P --> K["state/cache.py"]

    C --> SYS["core/system.py"]
    C --> S
    C --> T
    C --> M["state/memory.py"]
    C --> K
    C --> R

    I["__init__.py"] --> CFG["config/config.py"]
    I --> P
    I --> TESTS["tests/test.py (world.testing only)"]
```

## Runtime Request Flow

```mermaid
sequenceDiagram
    participant U as IRC User
    participant P as plugin.py
    participant C as core/core.py
    participant M as state/memory.py
    participant K as state/cache.py
    participant S as core/services.py
    participant G as Gemini API

    U->>P: gemini <query>
    P->>P: load_runtime_config() + capability checks
    P->>C: handle_query(...)
    C->>M: acquire_request_slot(...)
    C->>K: lookup(query, context)

    alt cache hit
        K-->>C: cached response
        C-->>P: final (optionally prefixed [cached])
    else cache miss
        C->>P: emit delayed progress indicator
        C->>S: generate_content(...)
        S->>G: models.generate_content(...)
        G-->>S: model response
        S-->>C: response payload
        C->>K: store(query, response, context)
        C-->>P: final cleaned/sanitized response
    end

    C->>M: release_request_slot(...)
    P-->>U: IRC reply
```

## Module Responsibilities

- `plugin.py`: command handlers (`gemini`, `gemversion`, `gemcache`) and minimal wiring.
- `core/core.py`: query lifecycle, capability enforcement, tool invocation, caching integration.
- `core/system.py`: tool schemas and system instruction constants.
- `core/services.py`: Gemini client creation and async execution boundary.
- `state/memory.py`: channel history and per-user/per-channel throttling state.
- `state/cache.py`: persistent response cache and fuzzy matching.
- `config/config.py`: persistent Limnoria registry settings declaration.
- `config/config_runtime.py`: converts registry values into typed runtime config.

## Architecture Rules (Enforced)

- Canonical imports must use package paths (`core.*`, `state.*`, `config.*`).
- Legacy flat module paths (for example `Geminoria.cache`) are removed in Phase 2.
- `plugin.py` stays thin; orchestration logic belongs in `core/core.py`.
- Architecture validation lives in `tests/test_architecture.py`.
