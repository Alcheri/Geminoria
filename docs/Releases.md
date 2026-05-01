<!-- Releases.md: Release and patch notes for Geminoria -->

## **_Agentic Gemini‑powered search for [Limnoria](https://github.com/progval/Limnoria) — Beta Release_**

This beta introduces a major architectural upgrade to Geminoria, shifting from environment‑based configuration to a fully **Limnoria‑native**, **config‑driven**, **agentic design**.

# ✨ What’s new

## Config‑driven Gemini integration

- Gemini API key now stored in Limnoria config: supybot.plugins.Geminoria.apiKey
- Model selection also moved to config: supybot.plugins.Geminoria.model
- Hot‑reload behaviour: client automatically refreshes when the key changes.

# Improved agentic loop

- More robust round‑based tool execution.
- Cleaner extraction of function calls and text responses.
- Final synthesis step added when tool‑call limit is reached.
- Graceful fallback to tool results if Gemini returns no text.

# Enhanced config search

- New config walker returns (full_path, is_leaf) pairs.
- Leaf keys are prioritised for relevance.
- Output formatted as a clean, readable list for IRC.
- Automatic highlighting of exact config keys in responses.

# Better logging

- Round‑by‑round agentic debugging.
- Tool call tracing with summarised output.
- Capability‑denial logging.
- Client refresh logging.
- Compact text summariser for log readability.

# Cleaner SDK integration

- Added _schema(),_tool(), and _gen_config() helpers to avoid Pylance issues with generated models.
- System instruction rewritten for clarity and IRC‑friendly output.

# 🛠️ Fixes & refinements

- More predictable whitespace and markdown cleaning.
- URL and message buffers remain lightweight and bounded.
- Capability checks now log denials for easier debugging.
- Safer handling of missing candidates or malformed responses.
- Improved fallback behaviour when Gemini returns empty content.

# ⚠️ Beta notes

- Geminoria is now published as a SemVer beta release (`1.1.0-beta.3`).
- Internal architecture now supports modular service/core/memory/cache separation.
- Please report any issues with tool-calling behaviour, config detection, or Gemini responses.

---

## 🌐 2026-04-19 - Network-Aware History Tool Allowlists (v1.1.0-beta.4)

- Bumped release version from `1.1.0-beta.3` to `1.1.0-beta.4`.
- Changed `historyToolsChannelAllowlist` to a network-scoped setting.
- Changed `searchLastChannelAllowlist` to a network-scoped setting.
- Changed `searchUrlsChannelAllowlist` to a network-scoped setting.
- Updated runtime tool gating to read those allowlists from the active IRC network context.
- Updated README examples to show explicit per-network config paths.

## ⏳ 2026-04-13 - Progress Indicator UX

- Added delayed, non-spam progress status for `@gemini` while Gemini is running.
- Default behavior is enabled and quiet:
  - `progressIndicatorEnabled=True`
  - `progressIndicatorDelayMs=1200`
  - Single status line only (no frame animation spam)
- Added configurable indicator presentation:
  - `progressIndicatorStyle` (`dots` or `plain`)
  - `progressIndicatorMessage` (custom status text override)
- Indicator is only used on non-cached runs and is emitted at most once per request.

## 🏷️ 2026-04-13 - SemVer Release Tag (v1.1.0-beta.3)

- Bumped release version from `1.1.0-beta.2` to `1.1.0-beta.3`.
- Updated package metadata and docs version tags to `1.1.0-beta.3`.
- Finalized architecture split baseline for future collaborative modules (`core.py`, `services.py`, `memory.py`, `cache.py`).

## 🔒 2026-04-05 - Security Hardening (v1.1.0-beta.2)

- Added sensitive-data redaction before sending user/tool payloads to Gemini (`redactSensitiveData`).
- Added safer logging defaults to avoid raw payload leakage (`logSensitiveData` defaults to `False`).
- Added anti-abuse controls:
  - Per-user cooldown (`cooldownSeconds`)
  - Per-channel concurrency cap (`maxConcurrentPerChannel`)
  - Reply length cap (`maxReplyChars`)
- Added output sanitization to strip IRC control characters before replies.
- Clarified docs and plugin help text around Limnoria capability behavior (default-allow unless anti-capabilities are configured).
- Added channel-level controls for history tools:
  - `allowSearchLast` (channel)
  - `allowSearchUrls` (channel)
- Added global allowlist controls for history tools:
  - Shared allowlist: `historyToolsChannelAllowlist`
  - Tool-specific allowlists: `searchLastChannelAllowlist`, `searchUrlsChannelAllowlist`
  - Tool-specific allowlists take precedence over shared allowlist when set.
- Added `TO-DO.md` tracking next security improvements.

## 🧹 2026-04-04 - Published Build Update

- Removed the `todo` command and all related storage/handler code from the GitHub-published Geminoria plugin.
- Removed `todo` documentation from README.
- Removed `todo`-specific tests from the test suite.
