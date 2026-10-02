# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Agent configuration discovery utilities for Garak's recon phase.

Standalone functions for loading and discovering a target's purpose and tool
surface. These are the building blocks for a harness-level recon phase that
runs before any probe is instantiated and places the result on the target
generator as ``target.capabilities``.

Current consumers: :class:`garak.probes.agent_breaker.AgentBreaker` (via thin
wrapper methods) and :class:`garak.harnesses.base.Harness` (recon phase).
"""

import datetime
import json
import logging
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

import yaml

import garak.attempt
from garak.exception import GarakException


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


# ---------------------------------------------------------------------------
# Generic ToolManifest builder (source-agnostic)
# ---------------------------------------------------------------------------

_SENSITIVE_KEYWORDS: List[str] = [
    "secret", "key", "token", "password", "credential",
    "private", "pii", "personal", "user data",
]
_MUTATE_KEYWORDS: List[str] = [
    "update", "delete", "write", "post", "create", "modify", "remove",
]
_EXEC_KEYWORDS: List[str] = [
    "execute", "run", "eval", "shell", "subprocess", "command",
]
_AUTH_KEYWORDS: List[str] = [
    "auth", "login", "impersonate", "on behalf", "credential",
]


def _keyword_match(text: str, keywords: List[str]) -> bool:
    t = text.lower()
    return any(kw in t for kw in keywords)


def _has_url_param(tool: dict) -> bool:
    """Return True if any input parameter looks like an external URL/endpoint."""
    props = tool.get("inputSchema", {}).get("properties", {})
    for prop in props.values():
        if prop.get("format") in ("uri", "url", "hostname"):
            return True
        desc = prop.get("description", "").lower()
        if any(kw in desc for kw in ("url", "endpoint", "webhook", "host")):
            return True
    return False


def _annotate_security(tool: dict) -> dict:
    """Compute security capability annotations for a single tool entry.

    Heuristics are keyed on description keyword matching and inputSchema
    structure — they work on any normalised tool dict, not just MCP ones.

    :param tool: Normalised tool dict with at minimum ``name``, ``description``,
        ``inputSchema``, and ``annotations`` keys.
    :returns: Dict for the ``security_annotations`` key in a ToolManifest tool entry.
    """
    annotations = tool.get("annotations", {})
    description = tool.get("description", "")

    capability_class: List[str] = []

    if annotations.get("openWorldHint") or _has_url_param(tool):
        capability_class.append("network_egress")
    if annotations.get("destructiveHint") or _keyword_match(description, _MUTATE_KEYWORDS):
        capability_class.append("write_mutate")
    if (
        annotations.get("destructiveHint")
        and not annotations.get("idempotentHint", True)
        and "irreversible" not in capability_class
    ):
        capability_class.append("irreversible")
    if _keyword_match(description, _SENSITIVE_KEYWORDS):
        capability_class.append("read_sensitive")
    if _keyword_match(description, _EXEC_KEYWORDS):
        capability_class.append("code_exec")
    if _keyword_match(description, _AUTH_KEYWORDS) and "auth_identity" not in capability_class:
        capability_class.append("auth_identity")

    is_source = "read_sensitive" in capability_class
    is_sink = (
        "network_egress" in capability_class and bool(annotations.get("openWorldHint"))
    ) or (
        "write_mutate" in capability_class and _has_url_param(tool)
    )

    if is_sink and not is_source:
        max_depth, allowed_successors = 0, []
    elif is_source:
        max_depth = 3
        allowed_successors = [
            c for c in ["write_mutate", "network_egress", "auth_identity"]
            if c not in capability_class
        ]
    else:
        max_depth, allowed_successors = 3, []

    return {
        "capability_class": capability_class,
        "is_source": is_source,
        "is_sink": is_sink,
        "chain_policy": {
            "max_depth": max_depth,
            "allowed_successors": allowed_successors,
        },
    }


def build_tool_manifest(server_info: dict, raw_tools: List[dict]) -> dict:
    """Build a ToolManifest (Layer 1 + Layer 2) from any normalised tool list.

    ``raw_tools`` is source-agnostic: it may come from MCP ``tools/list``,
    an OpenAPI spec conversion, a static YAML, or any other enumeration
    backend, as long as each entry carries ``name``, ``description``,
    ``inputSchema``, and ``annotations`` keys (all optional except ``name``).

    Layer 3 (Relay telemetry) is a stretch goal and is not populated here.

    :param server_info: Metadata about the tool source — at minimum
        ``endpoint`` and ``transport`` strings; callers may add any extra
        fields. Stored verbatim in the ``server`` block of the manifest.
    :param raw_tools: List of normalised tool dicts.
    :returns: ToolManifest dict matching the schema in the design doc.
    """
    tool_entries = []
    for t in raw_tools:
        entry: dict = {
            "name": t.get("name", ""),
            "description": t.get("description", ""),
            "inputSchema": t.get("inputSchema", {}),
            "annotations": t.get("annotations", {}),
            "security_annotations": _annotate_security(t),
        }
        tool_entries.append(entry)

    cc = [set(e["security_annotations"]["capability_class"]) for e in tool_entries]
    return {
        "server": {
            **server_info,
            "enumerated_at": datetime.datetime.now(datetime.UTC).isoformat(),
        },
        "capability_gate": {
            "mcp_tool_surface": bool(tool_entries),
            "tool_count": len(tool_entries),
            "has_destructive_tools": any(
                {"write_mutate", "irreversible"} & c for c in cc
            ),
            "has_external_egress": any("network_egress" in c for c in cc),
            "has_read_sensitive": any("read_sensitive" in c for c in cc),
        },
        "tools": tool_entries,
    }


# ---------------------------------------------------------------------------
# MCP enumeration source plugin
# ---------------------------------------------------------------------------


def mcp_enumerate(
    server_url: str,
    transport: str = "streamable-http",
    timeout: float = 30.0,
) -> List[dict]:
    """Issue a ``tools/list`` JSON-RPC call to an MCP server.

    Returns the raw tools list from the response — a list of dicts each
    carrying at minimum ``name`` and ``description``.  Pass the result
    directly to :func:`build_tool_manifest` together with a ``server_info``
    dict of your choosing.

    Only HTTP-based transports (``streamable-http``, ``http``) are supported.
    ``stdio`` agents must be handled by the caller.

    :param server_url: Full URL of the MCP endpoint.
    :param transport: Transport hint; only HTTP is implemented here.
    :param timeout: Request timeout in seconds.
    :returns: List of raw tool dicts from the server.
    :raises GarakException: On network error, non-2xx response, or invalid JSON.
    """
    if transport == "stdio":
        raise GarakException(
            "stdio MCP transport requires a running subprocess — "
            "call mcp_enumerate only for HTTP-based MCP servers."
        )

    payload = json.dumps(
        {"jsonrpc": "2.0", "method": "tools/list", "params": {}, "id": 1}
    ).encode()
    req = urllib.request.Request(
        server_url,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read()
    except urllib.error.HTTPError as exc:
        raise GarakException(
            f"MCP tools/list returned HTTP {exc.code}: {exc.reason}"
        ) from exc
    except urllib.error.URLError as exc:
        raise GarakException(f"MCP tools/list network error: {exc.reason}") from exc

    try:
        data = json.loads(body)
    except json.JSONDecodeError as exc:
        raise GarakException(
            f"MCP tools/list response is not valid JSON: {exc}"
        ) from exc

    # JSON-RPC 2.0 success: {"jsonrpc": "2.0", "result": {"tools": [...]}, "id": 1}
    if "error" in data:
        raise GarakException(f"MCP tools/list JSON-RPC error: {data['error']}")

    tools: List[dict] = data.get("result", {}).get("tools", [])
    logging.info("recon # MCP enumerated %d tools from %s", len(tools), server_url)
    return tools


# ---------------------------------------------------------------------------
# Capability container
# ---------------------------------------------------------------------------


@dataclass
class TargetCapabilities:
    """Encapsulates what the recon phase has learned about a target.

    Returned by :class:`ReconPlugin` implementations and accumulated by the
    harness before being placed on the target generator as
    ``generator.capabilities``.
    """

    purpose: str = ""
    tools: List[dict] = field(default_factory=list)
    # Full annotated manifest produced by MCPReconPlugin; None for other sources.
    tool_manifest: Optional[dict] = None

    def has_tools(self) -> bool:
        """Return True if at least one tool has been discovered."""
        return bool(self.tools)

    def merge(self, other: "TargetCapabilities") -> "TargetCapabilities":
        """Return a new TargetCapabilities combining self with other.

        *other*'s non-empty fields take precedence over self's, so later
        plugins in the priority list can refine earlier results.
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
# Recon plugin base and concrete implementations
# ---------------------------------------------------------------------------


class ReconPlugin:
    """Base class for recon source plugins.

    Each subclass encapsulates one way of gathering capability information
    about a target: from a static file, via MCP enumeration, or by asking
    the target agent directly.  The harness runs plugins in priority order
    and merges their results into a single :class:`TargetCapabilities`.
    """

    @classmethod
    def applicable(cls, recon_cfg: dict) -> bool:
        """Return True if *recon_cfg* contains enough config to attempt a run."""
        raise NotImplementedError

    def run(
        self,
        target,
        recon_cfg: dict,
        current: TargetCapabilities,
    ) -> Optional[TargetCapabilities]:
        """Gather capabilities about *target* and return the result.

        :param target: The target generator under test.
        :param recon_cfg: The ``recon`` config dict from the harness.
        :param current: Capabilities accumulated so far; use to avoid
            redundant work (e.g. skip discovery when tools already known).
        :returns: Newly gathered :class:`TargetCapabilities`, or ``None``
            when this plugin found nothing or chose to skip.
        """
        raise NotImplementedError


class FileReconPlugin(ReconPlugin):
    """Load target purpose and tools from a static YAML file."""

    @classmethod
    def applicable(cls, recon_cfg: dict) -> bool:
        return bool(recon_cfg.get("agent_config_path"))

    def run(
        self,
        target,
        recon_cfg: dict,
        current: TargetCapabilities,
    ) -> Optional[TargetCapabilities]:
        from garak.data import path as data_path

        config_path = recon_cfg["agent_config_path"]
        try:
            cfg = load_agent_config(data_path / config_path)
            return TargetCapabilities(
                purpose=cfg.get("agent_purpose", ""),
                tools=cfg.get("tools", []),
            )
        except Exception as exc:
            logging.warning("recon FileReconPlugin: %s", exc)
            return None


class MCPReconPlugin(ReconPlugin):
    """Enumerate tools from an MCP server via a tools/list JSON-RPC call."""

    @classmethod
    def applicable(cls, recon_cfg: dict) -> bool:
        return bool(recon_cfg.get("mcp", {}).get("server_url"))

    def run(
        self,
        target,
        recon_cfg: dict,
        current: TargetCapabilities,
    ) -> Optional[TargetCapabilities]:
        mcp_cfg = recon_cfg["mcp"]
        mcp_url = mcp_cfg["server_url"]
        mcp_transport = mcp_cfg.get("transport", "streamable-http")
        mcp_timeout = float(mcp_cfg.get("timeout", 30.0))
        try:
            raw_tools = mcp_enumerate(mcp_url, mcp_transport, mcp_timeout)
            tool_manifest = build_tool_manifest(
                {"endpoint": mcp_url, "transport": mcp_transport}, raw_tools
            )
            tools = [
                {
                    "name": t["name"],
                    "description": t["description"],
                    "inputSchema": t.get("inputSchema", {}),
                    "annotations": t.get("annotations", {}),
                }
                for t in tool_manifest["tools"]
            ]
            return TargetCapabilities(tools=tools, tool_manifest=tool_manifest)
        except Exception as exc:
            logging.warning("recon MCPReconPlugin: %s", exc)
            return None


class DiscoverReconPlugin(ReconPlugin):
    """Ask the target agent about its purpose and tools.

    Skipped automatically when the accumulated :class:`TargetCapabilities`
    already contains tools, to avoid unnecessary (and potentially slow)
    model calls.
    """

    @classmethod
    def applicable(cls, recon_cfg: dict) -> bool:
        return bool(recon_cfg.get("discover"))

    def run(
        self,
        target,
        recon_cfg: dict,
        current: TargetCapabilities,
    ) -> Optional[TargetCapabilities]:
        if current.has_tools():
            logging.info("recon DiscoverReconPlugin: tools already known, skipping")
            return None

        discover_cfg = recon_cfg["discover"]
        from garak.data import path as data_path
        import yaml

        prompts_path = data_path / discover_cfg.get(
            "prompts_path", "agent_breaker/prompts.yaml"
        )
        try:
            with open(prompts_path, "r", encoding="utf-8") as fh:
                prompts = yaml.safe_load(fh)
        except Exception as exc:
            logging.warning("recon DiscoverReconPlugin: failed to load prompts: %s", exc)
            return None

        parse_generator = self._load_parse_generator(discover_cfg)

        def _parse_fn(prompt: str) -> Optional[dict]:
            gen = parse_generator if parse_generator is not None else target
            conv = garak.attempt.Conversation(
                [
                    garak.attempt.Turn(
                        role="user",
                        content=garak.attempt.Message(text=prompt),
                    )
                ]
            )
            try:
                resp = gen.generate(prompt=conv, generations_this_call=1)
            except Exception as exc:
                logging.warning("recon DiscoverReconPlugin: parse call failed: %s", exc)
                return None
            if not resp or resp[0] is None or resp[0].text is None:
                return None
            return extract_json(resp[0].text)

        current_dict = {"agent_purpose": current.purpose, "tools": current.tools}
        try:
            updated = discover_agent_config(target, current_dict, prompts, _parse_fn)
        except Exception as exc:
            logging.warning("recon DiscoverReconPlugin: discovery failed: %s", exc)
            return None

        if updated is current_dict:
            return None
        return TargetCapabilities(
            purpose=updated.get("agent_purpose", ""),
            tools=updated.get("tools", []),
        )

    def _load_parse_generator(self, discover_cfg: dict):
        """Instantiate the parse generator when configured; returns None to use target."""
        import copy
        from garak import _plugins

        parse_model_type = discover_cfg.get("parse_model_type")
        if not parse_model_type:
            return None
        parse_model_name = discover_cfg.get("parse_model_name")
        parse_model_config = discover_cfg.get("parse_model_config") or {}
        try:
            generator_root: dict = {"generators": {}}
            conf_root = generator_root["generators"]
            for part in parse_model_type.split("."):
                conf_root = conf_root.setdefault(part, {})
            conf_root.update(copy.deepcopy(parse_model_config))
            if parse_model_name:
                conf_root["name"] = parse_model_name
            return _plugins.load_plugin(
                f"generators.{parse_model_type}",
                config_root=generator_root,
            )
        except Exception as exc:
            logging.warning(
                "recon DiscoverReconPlugin: failed to load parse generator: %s", exc
            )
            return None