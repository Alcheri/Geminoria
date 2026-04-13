# -*- coding: utf-8 -*-
"""System prompt and Gemini tool declaration helpers."""

from __future__ import annotations

from typing import Any

from google.genai import types as gtypes

SYSTEM_INSTRUCTION = (
    "You answer questions about a Limnoria bot using tool results. "
    "When the question is about configuration, prefer exact full config variable "
    "names first, then brief explanations. If tool results contain concrete keys, "
    "quote them verbatim and do not replace them with vague group summaries. "
    "Keep answers concise and readable for IRC."
)


def schema(**kwargs: Any) -> gtypes.Schema:
    return gtypes.Schema.model_validate(kwargs)


def tool(**kwargs: Any) -> gtypes.Tool:
    return gtypes.Tool.model_validate(kwargs)


def gen_config(**kwargs: Any) -> gtypes.GenerateContentConfig:
    return gtypes.GenerateContentConfig.model_validate(kwargs)


def make_tools(
    max_results: int, *, allow_search_last: bool, allow_search_urls: bool
) -> gtypes.Tool:
    n = int(max_results)
    declarations = [
        gtypes.FunctionDeclaration(
            name="search_config",
            description=(
                f"Search Limnoria's configuration registry for variables "
                f"whose name contains the given keyword. Returns up to {n} exact "
                f"full config variable names. When answering config questions, "
                f"cite these exact keys verbatim."
            ),
            parameters=schema(
                type=gtypes.Type.OBJECT,
                properties={
                    "word": schema(
                        type=gtypes.Type.STRING,
                        description="Keyword to search for in config variable names.",
                    )
                },
                required=["word"],
            ),
        ),
        gtypes.FunctionDeclaration(
            name="search_commands",
            description=(
                f"Search all loaded Limnoria plugin commands by name (like @apropos). "
                f"Returns up to {n} matching commands in 'Plugin.command' format."
            ),
            parameters=schema(
                type=gtypes.Type.OBJECT,
                properties={
                    "word": schema(
                        type=gtypes.Type.STRING,
                        description="Keyword to search for in command names.",
                    )
                },
                required=["word"],
            ),
        ),
    ]

    if allow_search_last:
        declarations.append(
            gtypes.FunctionDeclaration(
                name="search_last",
                description=(
                    f"Search recent channel messages for those containing the given text "
                    f"(like @last --with). Returns up to {n} results as 'nick: message'."
                ),
                parameters=schema(
                    type=gtypes.Type.OBJECT,
                    properties={
                        "text": schema(
                            type=gtypes.Type.STRING,
                            description="Text to search for inside recent messages.",
                        )
                    },
                    required=["text"],
                ),
            )
        )

    if allow_search_urls:
        declarations.append(
            gtypes.FunctionDeclaration(
                name="search_urls",
                description=(
                    f"Search recently posted channel URLs for those containing the given "
                    f"keyword (like @url search). Returns up to {n} results as 'nick: url'."
                ),
                parameters=schema(
                    type=gtypes.Type.OBJECT,
                    properties={
                        "word": schema(
                            type=gtypes.Type.STRING,
                            description="Keyword to search for in recent URLs.",
                        )
                    },
                    required=["word"],
                ),
            )
        )

    return tool(function_declarations=declarations)
