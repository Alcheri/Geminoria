# Code of Conduct

## Purpose

Geminoria is a Gemini-powered agentic search plugin for Limnoria. It can receive
IRC queries, call configured search tools, inspect recent channel history where
allowed, use a query cache, and return AI-generated answers.

This Code of Conduct sets expectations for respectful community behaviour and
responsible AI use around this project.

## Community Standards

Contributors, operators, and users are expected to:

- Treat other people with respect and patience.
- Give and accept technical feedback constructively.
- Avoid harassment, intimidation, personal attacks, and discriminatory language.
- Avoid publishing private information without explicit permission.
- Keep discussion focused on improving the plugin and its safe operation.

Unacceptable behaviour includes abuse, harassment, sustained disruption, and
conduct that would reasonably make other people feel unsafe or unwelcome.

## Responsible AI And Tool Use

AI output can be wrong, incomplete, biased, or unsafe. Contributors and
operators must not present Geminoria's replies as authoritative professional
advice or as a substitute for human judgement.

Because Geminoria is agentic, contributions should preserve safeguards around:

- tool declarations and tool-call limits;
- capability checks and channel-level policy;
- history-tool allowlists for `search_last` and `search_urls`;
- redaction before data is sent to Gemini;
- output sanitisation before replies reach IRC;
- cache keys that avoid cross-network or cross-channel leakage.

Changes that weaken these protections should explain the operational trade-off
and include appropriate tests or review notes.

## Privacy And Data Handling

Queries, selected tool results, and permitted history context may be sent to an
external AI provider. Operators should make that clear to their users and avoid
feeding sensitive personal data, secrets, credentials, private messages, or
confidential operational details into the plugin.

Contributors should keep sensitive logging disabled by default and avoid adding
logs that expose raw prompts, tool payloads, tokens, hostmasks, or private
channel content unless there is a clear debugging need and an explicit operator
setting.

## Safety, Moderation, And Abuse Prevention

Geminoria should not be used to harass people, automate abuse, bypass channel
rules, impersonate trusted users, generate malicious instructions, or mine
channel history outside the configured policy.

Safety-related reports may involve:

- capability or allowlist bypasses;
- history-tool access outside intended channels;
- output that exposes secrets or private content;
- prompt or tool patterns that produce unsafe replies;
- cache behaviour that reuses results in the wrong context.

Report security-sensitive issues privately as described in
[Security Policy](SECURITY.md). General conduct concerns may be reported to
@Alcheri on GitHub.

## Operator Responsibilities

Operators are responsible for configuring the plugin appropriately for their IRC
network and community. This includes access controls, history-tool policy, cache
settings, logging settings, rate limits, and provider API credentials.

Operators should review the relevant provider policies before enabling the
plugin in public or semi-public channels:

- [Google AI Gemini API Terms](https://ai.google.dev/gemini-api/terms)
- [Google Privacy Policy](https://policies.google.com/privacy)

## Enforcement

Project maintainers may remove comments, reject contributions, close issues,
block users, or decline support when behaviour conflicts with this Code of
Conduct or creates avoidable safety risk.

Enforcement should be proportionate, documented where practical, and mindful of
the privacy and security of people reporting concerns.

## Attribution

The community standards and enforcement structure are informed by the
[Contributor Covenant](https://www.contributor-covenant.org/), version 2.0, and
adapted for an AI-enabled Limnoria plugin.
