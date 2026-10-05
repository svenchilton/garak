# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Base classes and shared utilities for the garak recon plugin system.

Each concrete plugin lives in its own module inside this package.  The harness
discovers plugins at runtime by inspecting the package, so adding a new plugin
requires only a new module file — no changes to the harness are needed.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

import yaml

import garak.attempt
from garak.configurable import Configurable
from garak.exception import GarakException


# ---------------------------------------------------------------------------
# Standalone utility functions
# ---------------------------------------------------------------------------


def extract_json(text: str) -> Optional[dict]:
    """Extract a JSON object from model output that may include surrounding prose.

    Tries strict ``json.loads`` first, then falls back to finding the first
    ``{ … }`` block with ``re.DOTALL``.

    :param text: Raw text from a model response.
    :returns: Parsed dict, or ``None`` if no valid JSON object was found.
    """
    if text is None:
        return None
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    return None


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


def _keyword_match(text: str, keywords) -> bool:
    """Return True if any keyword appears in *text* (case-insensitive).

    Accepts both plain strings and ``str``-based enum members as keywords.
    """
    t = text.lower()
    return any((kw.value if hasattr(kw, "value") else kw) in t for kw in keywords)


# ---------------------------------------------------------------------------
# Typed capability schema
# ---------------------------------------------------------------------------


@dataclass
class SecurityAnnotations:
    """Security-relevant properties of a single tool, derived from its spec."""

    capability_class: List[str] = field(default_factory=list)
    is_source: bool = False
    is_sink: bool = False
    chain_policy: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "capability_class": self.capability_class,
            "is_source": self.is_source,
            "is_sink": self.is_sink,
            "chain_policy": self.chain_policy,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SecurityAnnotations":
        return cls(
            capability_class=d.get("capability_class", []),
            is_source=d.get("is_source", False),
            is_sink=d.get("is_sink", False),
            chain_policy=d.get("chain_policy", {}),
        )


@dataclass
class ToolDefinition:
    """Typed representation of a single tool discovered during the recon phase.

    Fields map to the normalised tool schema used throughout garak:

    * ``name`` / ``description`` — human-readable identity.
    * ``input_schema`` — JSON Schema for the tool's parameters (``inputSchema``
      in MCP wire format).
    * ``annotations`` — protocol-level hints (``openWorldHint``, etc.).
    * ``security_annotations`` — garak-derived security properties.
    """

    name: str = ""
    description: str = ""
    input_schema: dict = field(default_factory=dict)
    annotations: dict = field(default_factory=dict)
    security_annotations: SecurityAnnotations = field(
        default_factory=SecurityAnnotations
    )

    def to_dict(self) -> dict:
        """Return a plain dict representation for probe and legacy consumers."""
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema,
            "annotations": self.annotations,
            "security_annotations": self.security_annotations.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ToolDefinition":
        """Construct from the normalised tool dict used in manifests."""
        sa_raw = d.get("security_annotations", {})
        return cls(
            name=d.get("name", ""),
            description=d.get("description", ""),
            input_schema=d.get("inputSchema", d.get("input_schema", {})),
            annotations=d.get("annotations", {}),
            security_annotations=SecurityAnnotations.from_dict(sa_raw),
        )


@dataclass
class TargetCapabilities:
    """Encapsulates what the recon phase has learned about a target.

    Returned by :class:`ReconPlugin` implementations and accumulated by the
    harness before being placed on the target generator as
    ``generator.capabilities``.
    """

    purpose: str = ""
    tools: List[ToolDefinition] = field(default_factory=list)
    tool_manifest: Optional[dict] = None

    def has_tools(self) -> bool:
        """Return True if at least one tool has been discovered."""
        return bool(self.tools)

    def merge(self, other: "TargetCapabilities") -> "TargetCapabilities":
        """Return a new TargetCapabilities combining self with other.

        *other*'s non-empty fields take precedence over self's.
        """
        return TargetCapabilities(
            purpose=other.purpose or self.purpose,
            tools=other.tools if other.tools else self.tools,
            tool_manifest=(
                other.tool_manifest
                if other.tool_manifest is not None
                else self.tool_manifest
            ),
        )


# ---------------------------------------------------------------------------
# Plugin base class
# ---------------------------------------------------------------------------


class ReconPlugin(Configurable):
    """Base class for recon source plugins.

    Each subclass encapsulates one strategy for gathering capability
    information about a target.  The harness discovers concrete plugins at
    runtime by inspecting this package; adding a new plugin requires only a
    new module file.

    Plugins are instantiated with the full ``recon`` config dict.  Whether a
    plugin has the configuration it needs to operate is reported by
    :meth:`applicable`; the harness skips plugins that return ``False``.
    Results from the plugins that do run are merged into a single
    :class:`TargetCapabilities` object.
    """

    active = True
    priority: int = 50

    def __init__(self, recon_cfg: dict = None):
        self._recon_cfg = recon_cfg or {}

    def applicable(self) -> bool:
        """Return True if this plugin has the configuration it needs to run."""
        return False

    def run(
        self,
        target,
        current: TargetCapabilities,
    ) -> Optional[TargetCapabilities]:
        """Gather capabilities about *target* and return the result.

        :param target: The target generator under test.
        :param current: Capabilities accumulated so far; use to avoid
            redundant work (e.g. skip discovery when tools already known).
        :returns: Newly gathered :class:`TargetCapabilities`, or ``None``
            when this plugin found nothing or chose to skip.
        """
        raise NotImplementedError
