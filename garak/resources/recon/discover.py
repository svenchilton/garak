# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Discovery-based recon plugin — asks the target agent for its capabilities."""

import logging
from typing import Optional

import garak.attempt
from garak.resources.recon.base import (
    ReconPlugin,
    TargetCapabilities,
    ToolDefinition,
    discover_agent_config,
    extract_json,
)


class DiscoverReconPlugin(ReconPlugin):
    """Ask the target agent about its purpose and tools.

    Skipped automatically when the accumulated :class:`TargetCapabilities`
    already contains tools, to avoid unnecessary (and potentially slow)
    model calls.
    """

    priority = 30

    def __init__(self, recon_cfg: dict = None):
        super().__init__(recon_cfg)
        self._discover_cfg: Optional[dict] = self._recon_cfg.get("discover")

    def applicable(self) -> bool:
        return bool(self._discover_cfg)

    def run(
        self,
        target,
        current: TargetCapabilities,
    ) -> Optional[TargetCapabilities]:
        if current.has_tools():
            logging.info("recon DiscoverReconPlugin: tools already known, skipping")
            return None

        from garak.data import path as data_path
        import yaml

        prompts_path = data_path / self._discover_cfg.get(
            "prompts_path", "agent_breaker/prompts.yaml"
        )
        try:
            with open(prompts_path, "r", encoding="utf-8") as fh:
                prompts = yaml.safe_load(fh)
        except Exception as exc:
            logging.warning(
                "recon DiscoverReconPlugin: failed to load prompts: %s", exc
            )
            return None

        parse_generator = self._load_parse_generator()

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
                logging.warning(
                    "recon DiscoverReconPlugin: parse call failed: %s", exc
                )
                return None
            if not resp or resp[0] is None or resp[0].text is None:
                return None
            return extract_json(resp[0].text)

        current_dict = {"agent_purpose": current.purpose, "tools": []}
        try:
            updated = discover_agent_config(target, current_dict, prompts, _parse_fn)
        except Exception as exc:
            logging.warning("recon DiscoverReconPlugin: discovery failed: %s", exc)
            return None

        if updated is current_dict:
            return None

        tools = [
            ToolDefinition(
                name=t.get("name", ""),
                description=t.get("description", ""),
                input_schema=t.get("inputSchema", t.get("input_schema", {})),
                annotations=t.get("annotations", {}),
            )
            for t in updated.get("tools", [])
            if isinstance(t, dict)
        ]
        return TargetCapabilities(
            purpose=updated.get("agent_purpose", ""),
            tools=tools,
        )

    def _load_parse_generator(self):
        """Instantiate the parse generator when configured; returns None to use target."""
        import copy
        from garak import _plugins

        parse_model_type = self._discover_cfg.get("parse_model_type")
        if not parse_model_type:
            return None
        parse_model_name = self._discover_cfg.get("parse_model_name")
        parse_model_config = self._discover_cfg.get("parse_model_config") or {}
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
