# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""File-based recon plugin — loads agent config from a static YAML file."""

import logging
from typing import Optional

from garak.exception import GarakException
from garak.resources.recon.base import (
    ReconPlugin,
    SecurityAnnotations,
    TargetCapabilities,
    ToolDefinition,
    load_agent_config,
)


class FileReconPlugin(ReconPlugin):
    """Load target purpose and tools from a static YAML configuration file."""

    DEFAULT_PARAMS = {
        "agent_config_path": None,
    }
    priority = 10

    def __init__(self, config_root=None):
        super().__init__(config_root)

    def applicable(self) -> bool:
        return bool(self.agent_config_path)

    def run(
        self,
        target,
        current: TargetCapabilities,
    ) -> Optional[TargetCapabilities]:
        from garak.data import path as data_path

        try:
            from garak.resources.recon.mcp import _annotate_security

            cfg = load_agent_config(data_path / self.agent_config_path)
            tools = [
                ToolDefinition(
                    name=t.get("name", ""),
                    description=t.get("description", ""),
                    input_schema=t.get("inputSchema", t.get("input_schema", {})),
                    annotations=t.get("annotations", {}),
                    security_annotations=SecurityAnnotations.from_dict(
                        _annotate_security(t)
                    ),
                )
                for t in cfg.get("tools", [])
                if isinstance(t, dict)
            ]
            return TargetCapabilities(
                purpose=cfg.get("agent_purpose", ""),
                tools=tools,
            )
        except (GarakException, TypeError) as exc:
            logging.warning("recon FileReconPlugin: %s", exc)
            return None
