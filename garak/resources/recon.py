# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Agent configuration discovery utilities for Garak's recon phase.

Standalone functions for loading and discovering an agent's purpose and tool
surface. These are the building blocks for a harness-level recon phase that
runs before any probe is instantiated and places the result on the generator.

Current consumers: :class:`garak.probes.agent_breaker.AgentBreaker` (via thin
wrapper methods). Future consumer: the harness recon phase, which will call
these functions directly and inject the result as ``generator.tool_manifest``.
"""

import logging
from pathlib import Path
from typing import Callable, Optional

import yaml

import garak.attempt
from garak.exception import GarakException


def load_agent_config(config_file_path: Path) -> dict:
    """Load agent purpose and tools from a YAML configuration file.

    :param config_file_path: Absolute path to the agent config YAML.
    :returns: Dict with at minimum ``agent_purpose`` (str) and ``tools``
        (list) keys.
    :raises GarakException: If the file cannot be read or parsed.
    """
    try:
        with open(config_file_path, "r", encoding="utf-8") as f:
            agent_config = yaml.safe_load(f)
    except Exception as e:
        msg = f"Failed to load agent config from {config_file_path}: {e}"
        logging.error(msg)
        raise GarakException(msg) from e

    if not agent_config:
        agent_config = {}

    agent_config.setdefault("agent_purpose", "")
    agent_config.setdefault("tools", [])

    logging.info(
        "recon # Loaded agent config with %d tools", len(agent_config["tools"])
    )
    return agent_config


def discover_agent_config(
    generator,
    agent_config: dict,
    prompts: dict,
    parse_fn: Callable[[str], Optional[dict]],
) -> dict:
    """Ask the target agent for its purpose and/or tools, then parse.

    Only queries for what is missing in *agent_config*:

    * If ``agent_purpose`` is set but ``tools`` is empty, ask for tools only.
    * If both are missing, ask for purpose and tools.
    * If ``tools`` is already populated, returns *agent_config* unchanged.

    The discovery prompt is sent to the *target* agent (``generator``). The
    response is handed to ``parse_fn``, which is expected to run it through the
    red-team parse model and return a structured dict (or ``None`` on failure).

    :param generator: Garak generator for the target agent.
    :param agent_config: Current agent config dict (not mutated).
    :param prompts: Prompt templates dict; must contain ``DISCOVERY_FULL``,
        ``DISCOVERY_TOOLS_ONLY``, ``PARSE_FULL``, and ``PARSE_TOOLS_ONLY``.
    :param parse_fn: ``(prompt: str) -> Optional[dict]`` — runs the parse
        model on the given prompt and returns the extracted dict or ``None``.
    :returns: Updated agent config dict (new object if changes were made).
    """
    has_purpose = bool(agent_config.get("agent_purpose"))
    has_tools = bool(agent_config.get("tools"))

    if has_tools:
        return agent_config

    discovery_prompt = (
        prompts["DISCOVERY_TOOLS_ONLY"] if has_purpose else prompts["DISCOVERY_FULL"]
    )

    logging.info("recon # Discovering agent config from target agent...")

    conv = garak.attempt.Conversation(
        [
            garak.attempt.Turn(
                role="user",
                content=garak.attempt.Message(text=discovery_prompt),
            ),
        ]
    )
    try:
        response = generator.generate(prompt=conv, generations_this_call=1)
    except Exception as e:
        logging.warning("recon # Discovery call failed: %s", e)
        return agent_config

    if not response or response[0] is None or response[0].text is None:
        logging.warning("recon # Agent returned empty response during discovery")
        return agent_config

    agent_response: str = response[0].text

    parse_prompt = (
        prompts["PARSE_TOOLS_ONLY"].format(agent_response=agent_response)
        if has_purpose
        else prompts["PARSE_FULL"].format(agent_response=agent_response)
    )

    parsed = parse_fn(parse_prompt)
    if not parsed:
        logging.warning("recon # Parse model failed to parse discovery response")
        return agent_config

    updated = dict(agent_config)

    discovered_tools = parsed.get("tools", [])
    if discovered_tools:
        updated["tools"] = discovered_tools
        logging.info("recon # Discovered %d tools from agent", len(discovered_tools))

    if not has_purpose:
        discovered_purpose: str = parsed.get("agent_purpose", "")
        if discovered_purpose:
            updated["agent_purpose"] = discovered_purpose
            logging.info("recon # Discovered agent purpose from agent")

    return updated