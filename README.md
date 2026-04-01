<!---Geminoria 1.1.0-beta.1 – Gemini-powered agentic search plugin for Limnoria. --->

# Geminoria v1.1.0-beta.1

A Gemini-powered agentic search plugin for [Limnoria](https://github.com/progval/Limnoria).

[![Tests](https://github.com/Alcheri/Geminoria/actions/workflows/tests.yml/badge.svg)](https://github.com/Alcheri/Geminoria/actions/workflows/tests.yml) [![Lint](https://github.com/Alcheri/Geminoria/actions/workflows/lint.yml/badge.svg)](https://github.com/Alcheri/Geminoria/actions/workflows/lint.yml) [![CodeQL](https://github.com/Alcheri/Geminoria/actions/workflows/codeql.yml/badge.svg)](https://github.com/Alcheri/Geminoria/security/code-scanning)

## Description

`Geminoria` exposes a single IRC command, `gemini <query>`, backed by Google's
Gemini AI.  Gemini is given four tools drawn directly from Limnoria's own search
capabilities and can call any combination of them to answer a user's question:

| Tool | Equivalent Limnoria command |
| --- | --- |
| `search_config` | `@config search <word>` |
| `search_commands` | `@apropos <word>` |
| `search_last` | `@last --with <text>` |
| `search_urls` | `@url search <word>` |

## Requirements

```
google-genai==1.50.1
```

Install with:

```bash
pip install -r requirements.txt
```

The plugin currently pins `google-genai==1.50.1`, which is the known-good
version for the Python 3.11 Limnoria environments used by Puss and Borg.

## API Key And Model

Set the API key in Limnoria's config, the same way as `GoogleMaps` and `Weather`:

```
@config plugins.Geminoria.apiKey <your key>
```

The key is stored as a private registry value.

Set the model the same way:

```
@config plugins.Geminoria.model <model name>
```

This is a normal persistent Limnoria setting, so you can change models without
editing the plugin code.

## Access Control

All users must hold the `Geminoria` Limnoria capability to use the command.
Grant it with:

```
@admin capability add <nick> Geminoria
```

Change `supybot.plugins.Geminoria.requiredCapability` to `admin`, `owner`,
or `''` (everyone) to adjust the access level.

## Configuration

| Setting | Default | Description |
| --- | --- | --- |
| `apiKey` | `''` | Gemini API key |
| `model` | `gemini-3-flash-preview` | Persistent Gemini model setting |
| `requiredCapability` | `Geminoria` | Capability required to use `gemini` |
| `maxResults` | `5` | Max results returned per tool call |
| `bufferSize` | `50` | Recent messages/URLs to keep in memory per channel |
| `maxToolRounds` | `3` | Max agentic tool-call rounds before returning |
| `disableANSI` | `False` | Strip IRC colour/bold from replies |

## Usage

```
<you> @gemini what config options control flood protection?
<Borg> supybot.abuse.flood.command  supybot.abuse.flood.command.maximum ...
```

## Licence

See [LICENCE.md](LICENCE.md).
