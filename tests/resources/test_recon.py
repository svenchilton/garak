# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for garak.resources.recon — generic tool enumeration utilities."""

import json
from unittest.mock import MagicMock, patch

import pytest

from garak.exception import GarakException
from garak.resources.recon import (
    _annotate_security,
    _has_url_param,
    _keyword_match,
    build_tool_manifest,
    discover_agent_config,
    extract_json,
    load_agent_config,
    mcp_enumerate,
    DiscoverReconPlugin,
    FileReconPlugin,
    MCPReconPlugin,
    TargetCapabilities,
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
# _annotate_security
# ---------------------------------------------------------------------------


def test_annotate_security_network_egress_via_open_world():
    tool = {
        "description": "Fetch a webpage.",
        "annotations": {"openWorldHint": True, "destructiveHint": False, "idempotentHint": True},
        "inputSchema": {},
    }
    annotations = _annotate_security(tool)
    assert "network_egress" in annotations["capability_class"]
    assert annotations["is_sink"] is True


def test_annotate_security_read_sensitive():
    tool = {
        "description": "Read the user's auth token from the store.",
        "annotations": {},
        "inputSchema": {},
    }
    annotations = _annotate_security(tool)
    assert "read_sensitive" in annotations["capability_class"]
    assert annotations["is_source"] is True


def test_annotate_security_destructive_and_irreversible():
    tool = {
        "description": "Delete a file permanently.",
        "annotations": {"destructiveHint": True, "idempotentHint": False},
        "inputSchema": {},
    }
    annotations = _annotate_security(tool)
    assert "write_mutate" in annotations["capability_class"]
    assert "irreversible" in annotations["capability_class"]


def test_annotate_security_benign_tool():
    tool = {
        "description": "Return the current UTC timestamp.",
        "annotations": {"readOnlyHint": True},
        "inputSchema": {},
    }
    annotations = _annotate_security(tool)
    assert annotations["capability_class"] == []
    assert annotations["is_source"] is False
    assert annotations["is_sink"] is False
    assert "probe_relevance" not in annotations


def test_annotate_security_chain_policy_sink_only():
    tool = {
        "description": "Send an email to recipient.",
        "annotations": {"openWorldHint": True, "idempotentHint": False},
        "inputSchema": {"properties": {"to": {"type": "string", "format": "email"}}},
    }
    annotations = _annotate_security(tool)
    assert annotations["is_sink"] is True
    assert annotations["chain_policy"]["max_depth"] == 0
    assert annotations["chain_policy"]["allowed_successors"] == []


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


def test_build_tool_manifest_security_annotations_present():
    manifest = build_tool_manifest({"endpoint": "x", "transport": "http"}, [FETCH_TOOL])
    tool = manifest["tools"][0]
    assert "security_annotations" in tool
    assert "capability_class" in tool["security_annotations"]
    assert "probe_relevance" not in tool["security_annotations"]


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


# ---------------------------------------------------------------------------
# TargetCapabilities
# ---------------------------------------------------------------------------


def test_target_capabilities_defaults():
    caps = TargetCapabilities()
    assert caps.purpose == "", "default purpose should be empty"
    assert caps.tools == [], "default tools should be empty"
    assert caps.tool_manifest is None, "default tool_manifest should be None"


def test_target_capabilities_has_tools_false():
    assert not TargetCapabilities().has_tools(), "empty tools → has_tools() is False"


def test_target_capabilities_has_tools_true():
    caps = TargetCapabilities(tools=[{"name": "do_thing"}])
    assert caps.has_tools(), "non-empty tools → has_tools() is True"


def test_target_capabilities_merge_purpose_from_other():
    base = TargetCapabilities(purpose="old", tools=[])
    other = TargetCapabilities(purpose="new", tools=[])
    merged = base.merge(other)
    assert merged.purpose == "new", "merge should take other.purpose when non-empty"


def test_target_capabilities_merge_purpose_keeps_base_when_other_empty():
    base = TargetCapabilities(purpose="original", tools=[])
    other = TargetCapabilities(purpose="", tools=[])
    merged = base.merge(other)
    assert merged.purpose == "original", "merge should keep base.purpose when other.purpose is empty"


def test_target_capabilities_merge_tools_from_other():
    tools_a = [{"name": "a"}]
    tools_b = [{"name": "b"}]
    base = TargetCapabilities(tools=tools_a)
    other = TargetCapabilities(tools=tools_b)
    merged = base.merge(other)
    assert merged.tools == tools_b, "merge should take other.tools when non-empty"


def test_target_capabilities_merge_keeps_base_tools_when_other_empty():
    tools_a = [{"name": "a"}]
    base = TargetCapabilities(tools=tools_a)
    other = TargetCapabilities(tools=[])
    merged = base.merge(other)
    assert merged.tools == tools_a, "merge should keep base.tools when other.tools is empty"


def test_target_capabilities_merge_tool_manifest():
    manifest = {"tools": [], "server": {}}
    base = TargetCapabilities(tool_manifest=manifest)
    other = TargetCapabilities(tool_manifest=None)
    merged = base.merge(other)
    assert merged.tool_manifest == manifest, "merge should keep base.tool_manifest when other has None"

    other2 = TargetCapabilities(tool_manifest={"tools": [], "server": {"endpoint": "x"}})
    merged2 = base.merge(other2)
    assert merged2.tool_manifest == other2.tool_manifest, "merge should take other.tool_manifest when set"


# ---------------------------------------------------------------------------
# Plugin applicability
# ---------------------------------------------------------------------------


def test_file_plugin_applicable_with_path():
    assert FileReconPlugin.applicable({"agent_config_path": "some/path.yaml"})


def test_file_plugin_not_applicable_without_path():
    assert not FileReconPlugin.applicable({})


def test_mcp_plugin_applicable_with_server_url():
    assert MCPReconPlugin.applicable({"mcp": {"server_url": "http://agent:8080/mcp"}})


def test_mcp_plugin_not_applicable_without_mcp():
    assert not MCPReconPlugin.applicable({})


def test_mcp_plugin_not_applicable_without_server_url():
    assert not MCPReconPlugin.applicable({"mcp": {}})


def test_discover_plugin_applicable():
    assert DiscoverReconPlugin.applicable({"discover": {"parse_model_type": "nim"}})


def test_discover_plugin_not_applicable():
    assert not DiscoverReconPlugin.applicable({})


# ---------------------------------------------------------------------------
# FileReconPlugin.run
# ---------------------------------------------------------------------------


def test_file_plugin_run_success(tmp_path):
    cfg = tmp_path / "agent.yaml"
    cfg.write_text("agent_purpose: My bot\ntools:\n- name: fetch\n  description: Get data\n")

    with patch("garak.data.path", tmp_path):
        plugin = FileReconPlugin()
        result = plugin.run(None, {"agent_config_path": "agent.yaml"}, TargetCapabilities())

    assert result is not None, "FileReconPlugin should return TargetCapabilities on success"
    assert result.purpose == "My bot"
    assert result.tools[0]["name"] == "fetch"


def test_file_plugin_run_missing_file(tmp_path):
    with patch("garak.data.path", tmp_path):
        plugin = FileReconPlugin()
        result = plugin.run(None, {"agent_config_path": "nonexistent.yaml"}, TargetCapabilities())
    assert result is None, "FileReconPlugin should return None on missing file"


# ---------------------------------------------------------------------------
# MCPReconPlugin.run
# ---------------------------------------------------------------------------


def test_mcp_plugin_run_success():
    tools_payload = [{"name": "get_data", "description": "Fetch data.", "inputSchema": {}, "annotations": {}}]
    mock_response_body = json.dumps(
        {"jsonrpc": "2.0", "result": {"tools": tools_payload}, "id": 1}
    ).encode()
    mock_resp = MagicMock()
    mock_resp.read.return_value = mock_response_body
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)

    recon_cfg = {"mcp": {"server_url": "http://agent:8080/mcp"}}
    with patch("urllib.request.urlopen", return_value=mock_resp):
        result = MCPReconPlugin().run(None, recon_cfg, TargetCapabilities())

    assert result is not None, "MCPReconPlugin should return TargetCapabilities on success"
    assert result.has_tools(), "MCPReconPlugin result should have tools"
    assert result.tools[0]["name"] == "get_data"
    assert result.tool_manifest is not None


def test_mcp_plugin_run_network_error():
    import urllib.error
    recon_cfg = {"mcp": {"server_url": "http://agent:8080/mcp"}}
    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("refused")):
        result = MCPReconPlugin().run(None, recon_cfg, TargetCapabilities())
    assert result is None, "MCPReconPlugin should return None on network error"


# ---------------------------------------------------------------------------
# DiscoverReconPlugin.run — skips when tools already known
# ---------------------------------------------------------------------------


def test_discover_plugin_skips_when_tools_present():
    current = TargetCapabilities(tools=[{"name": "existing_tool"}])
    result = DiscoverReconPlugin().run(None, {"discover": {}}, current)
    assert result is None, "DiscoverReconPlugin should skip when tools are already known"
