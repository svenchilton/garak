# SPDX-FileCopyrightText: Portions Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Base harness

A harness coordinates running probes on a generator, running detectors on the
outputs, and evaluating the results.

This module includes the class Harness, which all `garak` harnesses must
inherit from.
"""

import importlib
import json
import logging
import types
from typing import List, Optional

import tqdm

from garak import _config
from garak import _plugins
from garak.configurable import Configurable
import garak.attempt
import garak.probes.base


def _initialize_runtime_services():
    """Initialize and validate runtime services required for a successful test"""

    from garak.exception import GarakException

    # TODO: this block may be gated in the future to ensure it is only run once. At this time
    # only one harness will execute per run so the output here is reasonable.
    service_names = ["langservice", "intentservice"]
    for service_name in service_names:
        logging.info("service import: " + service_name)
        service = importlib.import_module(f"garak.services.{service_name}")
        try:
            if service.enabled():
                symbol, msg = service.start_msg()
                if len(msg):
                    logging.info(msg)
                    print(f"{symbol} {msg}")
                service.load()
        except GarakException as e:
            logging.critical("❌ %s setup failed!" % service_name, exc_info=e)
            raise e


def _emit_plugin_cache_entry(*plugin_instances) -> None:
    snapshot = {}
    for plugin_instance in plugin_instances:
        if plugin_instance is None:
            continue
        classpath = (
            f"{plugin_instance.__class__.__module__}."
            f"{plugin_instance.__class__.__name__}"
        ).replace("garak.", "")
        category = classpath.split(".")[0]
        meta = _plugins.PluginCache.plugin_info(classpath)
        snapshot.setdefault(category, {})[classpath] = meta

    if not snapshot:
        return

    snapshot["version"] = garak.__version__
    _config.transient.reportfile.write(
        json.dumps(
            {
                "entry_type": "plugin_cache",
                "run": _config.transient.run_id,
                "plugin_cache": snapshot,
            },
            cls=_plugins.PluginEncoder,
            ensure_ascii=False,
        )
        + "\n"
    )


class Harness(Configurable):
    """Class to manage the whole process of probing, detecting and evaluating"""

    active = True
    # list of strings naming modules required but not explicitly in garak by default
    extra_dependency_names = []

    DEFAULT_PARAMS = {
        "strict_modality_match": False,
        # Optional recon phase config.  Set to a dict to enable; None disables.
        # Plugins run in order; each is skipped when its config key is absent.
        #
        #   agent_config_path (str): path relative to the garak data dir to
        #       load a static agent config YAML (agent_purpose + tools).
        #
        #   discover (dict): live discovery from the target generator.
        #       Keys: parse_model_type, parse_model_name, parse_model_config
        #       (same semantics as AgentBreaker's parse_model_* params);
        #       omit parse_model_* to use the target generator itself.
        #
        #   mcp (dict): MCP tool enumeration via tools/list JSON-RPC.
        #       Keys: server_url (str, required), transport (str, default
        #       "streamable-http"), timeout (float, default 30.0).
        #       Result is placed on target.capabilities (tool list) and
        #       target.tool_manifest (full manifest with security annotations).
        "recon": None,
    }

    def __init__(self, config_root=_config):
        self._load_config(config_root)

        _initialize_runtime_services()

        logging.info("harness init: %s", self)

    def _load_buffs(self, buff_names: List) -> None:
        """Instantiate specified buffs into global config

        Inheriting classes call _load_buffs in their run() methods. They then call
        garak.harness.base.Harness.run themselves, and so if _load_buffs() is called
        from this base class, we'll end up w/ inefficient reinstantiation of buff
        objects. If one wants to use buffs directly with this harness without
        subclassing, then call this method instance directly.

        Don't use this in the base class's run method, garak.harness.base.Harness.run;
        harnesses should be explicit about how they expect to deal with buffs.
        """

        _config.buffmanager.buffs = []
        for buff_name in buff_names:
            err_msg = None
            try:
                _config.buffmanager.buffs.append(_plugins.load_plugin(buff_name))
                logging.debug("loaded %s", buff_name)
            except ValueError as ve:
                err_msg = f"❌🦾 buff load error:❌ {ve}"
            except Exception as e:
                err_msg = f"❌🦾 failed to load buff {buff_name}:❌ {e}"
            finally:
                if err_msg is not None:
                    print(err_msg)
                    logging.warning(err_msg)
                    continue

    def _run_recon(self, target) -> None:
        """Optional harness-level recon phase.

        Runs before the probe loop. If ``self.recon`` config is present,
        populates ``target.capabilities`` so that probes can read it from the
        generator rather than performing their own discovery.

        Plugins executed (in order, each skipped if its config key is absent):

        1. **load** — reads a static agent-config YAML from
           ``recon.agent_config_path`` (relative to the garak data dir).
        2. **discover** — asks the target generator about its purpose and
           tools using the prompts in ``data/agent_breaker/prompts.yaml``.
           Requires ``recon.discover`` sub-config with ``parse_model_type``
           and ``parse_model_name``; if absent, live discovery is skipped.
        3. **mcp** — issues a ``tools/list`` JSON-RPC call to an MCP server
           and populates ``target.tool_manifest`` with the annotated result.
        """
        recon_cfg = getattr(self, "recon", None)
        if not recon_cfg:
            return

        from garak.data import path as data_path
        from garak.resources.recon import (
            discover_agent_config,
            extract_json,
            load_agent_config,
        )

        capabilities: dict = {"agent_purpose": "", "tools": []}

        # Plugin 1: load from disk
        config_path = recon_cfg.get("agent_config_path")
        if config_path:
            try:
                capabilities = load_agent_config(data_path / config_path)
            except Exception as e:
                logging.warning("harness recon: load failed: %s", e)

        # Plugin 2: live discovery (only when tools are still missing)
        discover_cfg = recon_cfg.get("discover")
        if not capabilities.get("tools") and discover_cfg:
            prompts_path = data_path / discover_cfg.get(
                "prompts_path", "agent_breaker/prompts.yaml"
            )
            try:
                import yaml

                with open(prompts_path, "r", encoding="utf-8") as fh:
                    prompts = yaml.safe_load(fh)
            except Exception as e:
                logging.warning("harness recon: failed to load prompts: %s", e)
                prompts = None

            if prompts:
                parse_model_type = discover_cfg.get("parse_model_type")
                parse_model_name = discover_cfg.get("parse_model_name")
                parse_model_config = discover_cfg.get("parse_model_config") or {}

                if parse_model_type:
                    try:
                        import copy

                        generator_root: dict = {"generators": {}}
                        conf_root = generator_root["generators"]
                        for part in parse_model_type.split("."):
                            conf_root = conf_root.setdefault(part, {})
                        conf_root.update(copy.deepcopy(parse_model_config))
                        if parse_model_name:
                            conf_root["name"] = parse_model_name
                        parse_generator = _plugins.load_plugin(
                            f"generators.{parse_model_type}",
                            config_root=generator_root,
                        )
                    except Exception as e:
                        logging.warning(
                            "harness recon: failed to load parse generator: %s", e
                        )
                        parse_generator = None
                else:
                    parse_generator = None

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
                        logging.warning("harness recon: parse call failed: %s", exc)
                        return None
                    if not resp or resp[0] is None or resp[0].text is None:
                        return None
                    return extract_json(resp[0].text)

                try:
                    capabilities = discover_agent_config(
                        target, capabilities, prompts, _parse_fn
                    )
                except Exception as e:
                    logging.warning("harness recon: discover failed: %s", e)

        # Plugin 3: MCP tool enumeration
        mcp_cfg = recon_cfg.get("mcp")
        if mcp_cfg:
            from garak.resources.recon import build_tool_manifest, mcp_enumerate

            mcp_url = mcp_cfg.get("server_url")
            mcp_transport = mcp_cfg.get("transport", "streamable-http")
            mcp_timeout = float(mcp_cfg.get("timeout", 30.0))
            if mcp_url:
                try:
                    raw_tools = mcp_enumerate(mcp_url, mcp_transport, mcp_timeout)
                    server_info = {
                        "endpoint": mcp_url,
                        "transport": mcp_transport,
                    }
                    tool_manifest = build_tool_manifest(server_info, raw_tools)
                    target.tool_manifest = tool_manifest
                    if tool_manifest["capability_gate"]["mcp_tool_surface"]:
                        capabilities["tools"] = [
                            {
                                "name": t["name"],
                                "description": t["description"],
                                "inputSchema": t.get("inputSchema", {}),
                                "annotations": t.get("annotations", {}),
                            }
                            for t in tool_manifest["tools"]
                        ]
                except Exception as e:
                    logging.warning("harness recon: MCP enumeration failed: %s", e)

        target.capabilities = capabilities
        logging.info(
            "harness recon: placed capabilities on target (%d tools)",
            len(capabilities.get("tools", [])),
        )

    def _start_run_hook(self):
        self._http_lib_user_agents = _config.get_http_lib_agents()
        _config.set_all_http_lib_agents(_config.run.user_agent)

    def _end_run_hook(self):
        _config.set_http_lib_agents(self._http_lib_user_agents)

    def _run_detector(self, probe_result_attempts, detector_instance) -> None:
        logging.debug("harness: run detector %s", detector_instance.detectorname)
        attempt_iterator = tqdm.tqdm(probe_result_attempts, leave=False)
        detector_probe_name = detector_instance.detectorname.replace(
            "garak.detectors.", ""
        )
        # include the probe name in the detector progress bar so long runs
        # show which probe's results are being scored (#324); attempts carry
        # the probe classname already, so no caller-side plumbing is needed
        if len(probe_result_attempts) > 0:
            probe_display_name = probe_result_attempts[0].probe_classname
            attempt_iterator.set_description(
                f"{probe_display_name}/{detector_probe_name}"
            )
        else:
            attempt_iterator.set_description("detectors." + detector_probe_name)
        for attempt in attempt_iterator:
            if detector_instance.skip:
                continue
            attempt.detector_results[detector_probe_name] = list(
                detector_instance.detect(attempt)
            )

    def run(self, model, probes, detectors, evaluator, announce_probe=True) -> None:
        """Core harness method

        :param model: an instantiated generator providing an interface to the model to be examined
        :type model: garak.generators.Generator
        :param probes: a list of probe instances to be run
        :type probes: List[garak.probes.base.Probe]
        :param detectors: a list of detectors to use on the results of the probes
        :type detectors: List[garak.detectors.base.Detector]
        :param evaluator: an instantiated evaluator for judging detector results
        :type evaluator: garak.evaluators.base.Evaluator
        :param announce_probe: Should we print probe loading messages?
        :type announce_probe: bool, optional
        """
        if not detectors:
            msg = "No detectors, nothing to do"
            logging.warning(msg)
            if hasattr(_config.system, "verbose") and _config.system.verbose >= 2:
                print(msg)
            raise ValueError(msg)

        if not probes:
            msg = "No probes, nothing to do"
            logging.warning(msg)
            if hasattr(_config.system, "verbose") and _config.system.verbose >= 2:
                print(msg)
            raise ValueError(msg)

        self._start_run_hook()
        self._run_recon(model)
        _emit_plugin_cache_entry(
            self,
            model,
            *probes,
            *detectors,
            *_config.buffmanager.buffs,
        )

        for probe in probes:
            logging.debug("harness: probe start for %s", probe.probename)
            if not probe:
                continue

            modality_match = _modality_match(
                probe.modality["in"], model.modality["in"], self.strict_modality_match
            )

            if not modality_match:
                logging.warning(
                    "probe skipped due to modality mismatch: %s - model expects %s",
                    probe.probename,
                    model.modality["in"],
                )
                continue

            attempt_results = probe.probe(model)
            assert isinstance(
                attempt_results, (list, types.GeneratorType)
            ), "probing should always return an ordered iterable"

            if not isinstance(probe, garak.probes.base.IntentProbe):
                for d in detectors:
                    self._run_detector(attempt_results, d)

            else:
                # extract detectors to be run
                detectors_required = set()
                intents_observed = set()
                # determine candidate detectors
                attempt_results_list = list(attempt_results)
                intent_to_detector = {}
                probe_detector_names = {
                    d.detectorname.replace("garak.detectors.", "") for d in detectors
                }

                for a in attempt_results_list:
                    intent = a.intent
                    if not intent:
                        logging.warning(
                            "probe %s attempt %s seq %s has no or empty intent"
                            % (probe.probename, a.uuid, a.seq)
                        )
                    intents_observed.add(intent)

                if intents_observed:
                    from garak.services import intentservice

                for intent_observed in intents_observed:
                    detectors = intentservice.get_detectors(intent_observed)
                    if detectors is None:
                        logging.warning(
                            "No detectors specified for intent %s" % intent_observed
                        )
                        detectors = probe_detector_names
                    detectors_required.update(detectors)
                    intent_to_detector[intent_observed] = detectors

                logging.info(
                    "For probe %s, selected detectors %s based on intents"
                    % (probe.probename, repr(detectors_required))
                )

                intent_detectors = []
                for detector_name in detectors_required:
                    d = _plugins.load_plugin(f"detectors.{detector_name}")
                    intent_detectors.append(d)
                    attempt_subset = []
                    for a in attempt_results_list:
                        mapping = intent_to_detector[a.intent]
                        if detector_name in mapping:
                            attempt_subset.append(a)
                    self._run_detector(attempt_subset, d)

                # detectors resolved via the intent path are not in the
                # harness-level detector list snapshotted at run start, so emit
                # them here to keep report.jsonl plugin_cache complete
                _emit_plugin_cache_entry(*intent_detectors)
                del intent_detectors

            for attempt in attempt_results:
                attempt.status = garak.attempt.ATTEMPT_COMPLETE
                _config.transient.reportfile.write(
                    json.dumps(attempt.as_dict(), ensure_ascii=False) + "\n"
                )

            if len(attempt_results) == 0:
                logging.warning("zero attempt results: probe %s" % probe.probename)

            evaluator.evaluate(attempt_results)

        self._end_run_hook()

        logging.debug("harness: probe list iteration completed")


def _modality_match(probe_modality, generator_modality, strict):
    if strict:
        # must be perfect match
        return probe_modality == generator_modality
    else:
        # everything probe wants must be accepted by model
        return set(probe_modality).intersection(generator_modality) == set(
            probe_modality
        )
