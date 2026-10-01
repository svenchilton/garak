# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for garak.resources.recon — generic tool enumeration utilities."""

import json
from unittest.mock import MagicMock, patch

import pytest

from garak.exception import GarakException
from garak.resources.recon import (
    _compute_layer2,
    _has_url_param,
    _keyword_match,
    build_tool_manifest,
    discover_agent_config,
    extract_json,
    load_agent_config,
    mcp_enumerate,
)


# ---------------------------------------------------------------------------
# extract_json
# ---------------------------------------------------------------------------


def test_extract_json_plain():
    assert extract_json('{"a": 1}') == {"a": 1}


def test_extract_json_embedded():
    assert extract_json('Here is the result: {"tools": []} done.') == {"tools": []}


def test_extract_json_invalid():
    assert extract_json("no json here") is None


def test_extract_json_none():
    assert extract_json(None) is None


# ---------------------------------------------------------------------------
# _keyword_match / _has_url_param
# ---------------------------------------------------------------------------


def test_keyword_match_hit():
    assert _keyword_match("Retrieve secret key from vault", ["secret", "key"])


def test_keyword_match_miss():
    assert not _keyword_match("List calendar events", ["secret", "credential"])


def test_has_url_param_uri_format():
    tool = {"inputSchema": {"properties": {"target": {"type": "string", "format": "uri"}}}}
    assert _has_url_param(tool)


def test_has_url_param_description_hint():
    tool = {"inputSchema": {"properties": {"dest": {"type": "string", "description": "webhook url"}}}}
    assert _has_url_param(tool)


def test_has_url_param_none():
    tool = {"inputSchema": {"properties": {"name": {"type": "string"}}}}
    assert not _has_url_param(tool)


# ---------------------------------------------------------------------------
# _compute_layer2
# ---------------------------------------------------------------------------


def test_layer2_network_egress_via_open_world():
    tool = {
        "description": "Fetch a webpage.",
        "annotations": {"openWorldHint": True, "destructiveHint": False, "idempotentHint": True},
        "inputSchema": {},
    }
    layer2 = _compute_layer2(tool)
    assert "network_egress" in layer2["capability_class"]
    assert layer2["is_sink"] is True
    assert layer2["probe_relevance"]["A1_ipi_via_tool_results"] is True


def test_layer2_read_sensitive():
    tool = {
        "description": "Read the user's auth token from the store.",
        "annotations": {},
        "inputSchema": {},
    }
    layer2 = _compute_layer2(tool)
    assert "read_sensitive" in layer2["capability_class"]
    assert layer2["is_source"] is True


def test_layer2_destructive_and_irreversible():
    tool = {
        "description": "Delete a file permanently.",
        "annotations": {"destructiveHint": True, "idempotentHint": False},
        "inputSchema": {},
    }
    layer2 = _compute_layer2(tool)
    assert "write_mutate" in layer2["capability_class"]
    assert "irreversible" in layer2["capability_class"]
    assert layer2["probe_relevance"]["A2_permission_escalation"] is True


def test_layer2_benign_tool():
    tool = {
        "description": "Return the current UTC timestamp.",
        "annotations": {"readOnlyHint": True},
        "inputSchema": {},
    }
    layer2 = _compute_layer2(tool)
    assert layer2["capability_class"] == []
    assert layer2["is_source"] is False
    assert layer2["is_sink"] is False
    assert layer2["probe_relevance"]["A5_tool_metadata_poisoning"] is True


def test_layer2_chain_policy_sink_only():
    tool = {
        "description": "Send an email to recipient.",
        "annotations": {"openWorldHint": True, "idempotentHint": False},
        "inputSchema": {"properties": {"to": {"type": "string", "format": "email"}}},
    }
    layer2 = _compute_layer2(tool)
    assert layer2["is_sink"] is True
    assert layer2["chain_policy"]["max_depth"] == 0
    assert layer2["chain_policy"]["allowed_successors"] == []


# ---------------------------------------------------------------------------
# build_tool_manifest
# ---------------------------------------------------------------------------


FETCH_TOOL = {
    "name": "fetch_document",
    "description": "Retrieve a document by URL.",
    "inputSchema": {"properties": {"url": {"type": "string", "format": "uri"}}},
    "annotations": {"openWorldHint": True, "destructiveHint": False, "idempotentHint": True},
}

EMAIL_TOOL = {
    "name": "send_email",
    "description": "Send an email to a recipient.",
    "inputSchema": {"properties": {"to": {"type": "string", "format": "email"}}},
    "annotations": {"openWorldHint": True, "destructiveHint": False, "idempotentHint": False},
}


def test_build_tool_manifest_structure():
    server_info = {"endpoint": "http://agent:8080/mcp", "transport": "streamable-http"}
    manifest = build_tool_manifest(server_info, [FETCH_TOOL, EMAIL_TOOL])

    assert manifest["server"]["endpoint"] == "http://agent:8080/mcp"
    assert "enumerated_at" in manifest["server"]
    assert manifest["capability_gate"]["tool_count"] == 2
    assert manifest["capability_gate"]["mcp_tool_surface"] is True
    assert manifest["capability_gate"]["has_external_egress"] is True
    assert len(manifest["tools"]) == 2


def test_build_tool_manifest_layer2_present():
    manifest = build_tool_manifest({"endpoint": "x", "transport": "http"}, [FETCH_TOOL])
    tool = manifest["tools"][0]
    assert "garak" in tool
    assert "capability_class" in tool["garak"]
    assert "probe_relevance" in tool["garak"]


def test_build_tool_manifest_empty():
    manifest = build_tool_manifest({"endpoint": "x", "transport": "http"}, [])
    assert manifest["capability_gate"]["mcp_tool_surface"] is False
    assert manifest["capability_gate"]["tool_count"] == 0


def test_build_tool_manifest_preserves_server_info():
    server_info = {"endpoint": "http://x", "transport": "http", "protocol_version": "2024-11-05"}
    manifest = build_tool_manifest(server_info, [])
    assert manifest["server"]["protocol_version"] == "2024-11-05"


# ---------------------------------------------------------------------------
# mcp_enumerate
# ---------------------------------------------------------------------------


def test_mcp_enumerate_stdio_raises():
    with pytest.raises(GarakException, match="stdio"):
        mcp_enumerate("http://unused", transport="stdio")


def test_mcp_enumerate_success():
    tools_payload = [{"name": "get_data", "description": "Fetch data."}]
    mock_response_body = json.dumps(
        {"jsonrpc": "2.0", "result": {"tools": tools_payload}, "id": 1}
    ).encode()

    mock_resp = MagicMock()
    mock_resp.read.return_value = mock_response_body
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)

    with patch("urllib.request.urlopen", return_value=mock_resp):
        result = mcp_enumerate("http://agent:8080/mcp")

    assert result == tools_payload


def test_mcp_enumerate_jsonrpc_error():
    mock_response_body = json.dumps(
        {"jsonrpc": "2.0", "error": {"code": -32601, "message": "Method not found"}, "id": 1}
    ).encode()
    mock_resp = MagicMock()
    mock_resp.read.return_value = mock_response_body
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)

    with patch("urllib.request.urlopen", return_value=mock_resp):
        with pytest.raises(GarakException, match="JSON-RPC error"):
            mcp_enumerate("http://agent:8080/mcp")


def test_mcp_enumerate_http_error():
    import urllib.error

    with patch(
        "urllib.request.urlopen",
        side_effect=urllib.error.HTTPError("url", 404, "Not Found", {}, None),
    ):
        with pytest.raises(GarakException, match="HTTP 404"):
            mcp_enumerate("http://agent:8080/mcp")


def test_mcp_enumerate_network_error():
    import urllib.error

    with patch(
        "urllib.request.urlopen",
        side_effect=urllib.error.URLError("Connection refused"),
    ):
        with pytest.raises(GarakException, match="network error"):
            mcp_enumerate("http://agent:8080/mcp")


# ---------------------------------------------------------------------------
# load_agent_config
# ---------------------------------------------------------------------------


def test_load_agent_config_defaults(tmp_path):
    cfg = tmp_path / "agent.yaml"
    cfg.write_text("agent_purpose: Test agent\n")
    result = load_agent_config(cfg)
    assert result["agent_purpose"] == "Test agent"
    assert result["tools"] == []


def test_load_agent_config_empty_file(tmp_path):
    cfg = tmp_path / "empty.yaml"
    cfg.write_text("")
    result = load_agent_config(cfg)
    assert result == {"agent_purpose": "", "tools": []}


def test_load_agent_config_missing_raises(tmp_path):
    with pytest.raises(GarakException):
        load_agent_config(tmp_path / "nonexistent.yaml")


# ---------------------------------------------------------------------------
# discover_agent_config
# ---------------------------------------------------------------------------


def test_discover_skips_when_tools_present():
    config = {"agent_purpose": "Helper", "tools": [{"name": "tool_a"}]}
    result = discover_agent_config(None, config, {}, lambda _: None)
    assert result is config  # unchanged, parse_fn never called


def test_discover_updates_tools():
    config = {"agent_purpose": "", "tools": []}

    mock_gen = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "agent response"
    mock_gen.generate.return_value = [mock_response]

    prompts = {
        "DISCOVERY_FULL": "What tools do you have?",
        "PARSE_FULL": "Parse this: {agent_response}",
    }
    discovered = [{"name": "tool_x", "description": "Does X."}]
    parse_fn = lambda _: {"agent_purpose": "A helper", "tools": discovered}

    result = discover_agent_config(mock_gen, config, prompts, parse_fn)
    assert result["tools"] == discovered
    assert result["agent_purpose"] == "A helper"


def test_discover_returns_unchanged_on_empty_response():
    config = {"agent_purpose": "", "tools": []}
    mock_gen = MagicMock()
    mock_gen.generate.return_value = [MagicMock(text=None)]
    prompts = {"DISCOVERY_FULL": "?", "PARSE_FULL": "{agent_response}"}

    result = discover_agent_config(mock_gen, config, prompts, lambda _: None)
    assert result["tools"] == []
