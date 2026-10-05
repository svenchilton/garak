# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""garak.resources.recon — plugin-based recon framework for agentic targets.

Public symbols are re-exported here for backward compatibility; code that
previously imported from ``garak.resources.recon`` (when it was a single
module) continues to work without changes.
"""

from garak.resources.recon.base import (
    ReconPlugin,
    SecurityAnnotations,
    TargetCapabilities,
    ToolDefinition,
    _keyword_match,
    discover_agent_config,
    extract_json,
    load_agent_config,
)
from garak.resources.recon.discover import DiscoverReconPlugin
from garak.resources.recon.file import FileReconPlugin
from garak.resources.recon.mcp import (
    AuthKeyword,
    ExecKeyword,
    MCPReconPlugin,
    MutateKeyword,
    SensitiveKeyword,
    _annotate_security,
    _has_url_param,
    build_tool_manifest,
    mcp_enumerate,
)

__all__ = [
    # Base
    "ReconPlugin",
    "SecurityAnnotations",
    "TargetCapabilities",
    "ToolDefinition",
    "_keyword_match",
    "discover_agent_config",
    "extract_json",
    "load_agent_config",
    # Plugins
    "DiscoverReconPlugin",
    "FileReconPlugin",
    "MCPReconPlugin",
    # MCP helpers & enums
    "AuthKeyword",
    "ExecKeyword",
    "MutateKeyword",
    "SensitiveKeyword",
    "_annotate_security",
    "_has_url_param",
    "build_tool_manifest",
    "mcp_enumerate",
]
