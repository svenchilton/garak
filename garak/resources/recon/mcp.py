# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MCP recon plugin — tool enumeration via the MCP ``tools/list`` JSON-RPC call.

Keyword enums, security annotation helpers, the ``mcp_enumerate`` function,
``build_tool_manifest``, and the :class:`MCPReconPlugin` all live here because
they are MCP protocol concepts, not generic recon concepts.
"""

import datetime
import json
import logging
import urllib.error
import urllib.request
from enum import Enum
from typing import List, Optional

from garak.exception import GarakException
from garak.resources.recon.base import (
    ReconPlugin,
    SecurityAnnotations,
    TargetCapabilities,
    ToolDefinition,
    _keyword_match,
)


# ---------------------------------------------------------------------------
# Keyword enums
# ---------------------------------------------------------------------------


class SensitiveKeyword(str, Enum):
    SECRET = "secret"
    KEY = "key"
    TOKEN = "token"
    PASSWORD = "password"
    CREDENTIAL = "credential"
    PRIVATE = "private"
    PII = "pii"
    PERSONAL = "personal"
    USER_DATA = "user data"


class MutateKeyword(str, Enum):
    UPDATE = "update"
    DELETE = "delete"
    WRITE = "write"
    POST = "post"
    CREATE = "create"
    MODIFY = "modify"
    REMOVE = "remove"


class ExecKeyword(str, Enum):
    EVAL = "eval"
    SHELL = "shell"
    SUBPROCESS = "subprocess"
    COMMAND = "command"
    SCRIPT = "script"
    BASH = "bash"


class AuthKeyword(str, Enum):
    AUTH = "auth"
    LOGIN = "login"
    IMPERSONATE = "impersonate"
    ON_BEHALF = "on behalf"
    CREDENTIAL = "credential"


# ---------------------------------------------------------------------------
# Security annotation helpers
# ---------------------------------------------------------------------------


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
    if annotations.get("destructiveHint") or _keyword_match(description, MutateKeyword):
        capability_class.append("write_mutate")
    if (
        annotations.get("destructiveHint")
        and not annotations.get("idempotentHint", True)
        and "irreversible" not in capability_class
    ):
        capability_class.append("irreversible")
    if _keyword_match(description, SensitiveKeyword):
        capability_class.append("read_sensitive")
    if _keyword_match(description, ExecKeyword):
        capability_class.append("code_exec")
    if _keyword_match(description, AuthKeyword) and "auth_identity" not in capability_class:
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


# ---------------------------------------------------------------------------
# ToolManifest builder
# ---------------------------------------------------------------------------


def build_tool_manifest(server_info: dict, raw_tools: List[dict]) -> dict:
    """Build a ToolManifest (Layer 1 + Layer 2) from any normalised tool list.

    :param server_info: Metadata about the tool source — at minimum ``endpoint``
        and ``transport`` strings; extra fields are stored verbatim.
    :param raw_tools: List of normalised tool dicts.
    :returns: ToolManifest dict.
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
# MCP enumeration
# ---------------------------------------------------------------------------


def mcp_enumerate(
    server_url: str,
    transport: str = "streamable-http",
    timeout: float = 30.0,
) -> List[dict]:
    """Issue a ``tools/list`` JSON-RPC call to an MCP server.

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

    if "error" in data:
        raise GarakException(f"MCP tools/list JSON-RPC error: {data['error']}")

    tools: List[dict] = data.get("result", {}).get("tools", [])
    logging.info("recon # MCP enumerated %d tools from %s", len(tools), server_url)
    return tools


# ---------------------------------------------------------------------------
# Plugin
# ---------------------------------------------------------------------------


class MCPReconPlugin(ReconPlugin):
    """Enumerate tools from an MCP server via a tools/list JSON-RPC call."""

    priority = 20

    def __init__(self, recon_cfg: dict = None):
        super().__init__(recon_cfg)
        mcp = self._recon_cfg.get("mcp", {})
        self._server_url: Optional[str] = mcp.get("server_url")
        self._transport: str = mcp.get("transport", "streamable-http")
        self._timeout: float = float(mcp.get("timeout", 30.0))

    def applicable(self) -> bool:
        return bool(self._server_url)

    def run(
        self,
        target,
        current: TargetCapabilities,
    ) -> Optional[TargetCapabilities]:
        try:
            raw_tools = mcp_enumerate(self._server_url, self._transport, self._timeout)
            tool_manifest = build_tool_manifest(
                {"endpoint": self._server_url, "transport": self._transport},
                raw_tools,
            )
            tools = [ToolDefinition.from_dict(t) for t in tool_manifest["tools"]]
            return TargetCapabilities(tools=tools, tool_manifest=tool_manifest)
        except (GarakException, urllib.error.HTTPError, urllib.error.URLError, json.JSONDecodeError) as exc:
            logging.warning("recon MCPReconPlugin: %s", exc)
            return None
