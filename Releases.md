<!-- Releases.md: Release and patch notes for Geminoria -->

## **_Agentic Gemini‑powered search for [Limnoria](https://github.com/progval/Limnoria) — Beta Release_**

This beta introduces a major architectural upgrade to Geminoria, shifting from environment‑based configuration to a fully **Limnoria‑native**, **config‑driven**, **agentic design**. It is feature‑complete for the 1.1 series and ready for wider testing.

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

- This is a pre‑release intended for real‑world testing.
- Behaviour may change slightly before the final 1.1.0 release.
- Please report any issues with tool‑calling behaviour, config detection, or Gemini responses.

---

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
