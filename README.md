<!-- release-sync:title -->
# WSP
<!-- release-sync:title -->

<!-- release-sync:overview -->
Forward-only model-editing core with instance-scoped reviewed-candidate registries.
<!-- release-sync:overview -->

<!-- release-sync:capabilities -->
- Load a reviewed-candidate registry from an explicit JSON file
- Validate write policies against the instance's own registry
- Thread one registry per store through create/bind/dispatch (two stores in one thread keep disjoint registries)
<!-- release-sync:capabilities -->

<!-- release-sync:install -->
1. Requires Python 3.12+
2. Built against the pinned public upstream commit openwsp/openwsp@8d09276736034f6874bb40e7019247c9541a5551 (https://github.com/openwsp/openwsp/commit/8d09276736034f6874bb40e7019247c9541a5551)
3. Copy the src/ tree onto PYTHONPATH
4. Prepare a registry JSON (synthetic example: configs/example-registry.json)
<!-- release-sync:install -->

<!-- release-sync:configuration -->
- `registry_file`: path to this instance's reviewed-candidate registry JSON
- `scope`: public run-scope identifier; `single_fact_postfreeze_poc` declares the bounded post-freeze POC scope
- `write_policy.version`: policy version enum: `fixed_rank_one_v1`, `fixed_rank_one_branches_v2`, `fixed_two_stage_v3`, `fixed_mlp_gates_v4`; the registry must contain the reviewed entry for the selected version
- `isolation.mode`: `remote_worker_only_not_windows_editor_isolation` — the isolation evidence boundary covers the remote worker only; the Windows-side editor is outside this scope
- `upstream.use_reference_exp8_driver`: must be False — the reference exp8 driver is disabled; protocol validation rejects any protocol that enables it
- `upstream.use_pinned_codex_editor_adapter`: must be False — the pinned Codex editor adapter is disabled; protocol validation rejects any protocol that enables it
<!-- release-sync:configuration -->

<!-- release-sync:demo -->
- `python scripts/tiny_demo.py --run-dir demo-tiny` — run the full tiny store lifecycle: init, read, forward, patch, diff, rollback, export, reload, finish — every step asserted; standard-library only, no external packages, model or GPU required
<!-- release-sync:demo -->

<!-- release-sync:alternatives -->
- **Reviewed-candidate registry**: loaded per instance from explicit config; no global registry
- **Cross-layer engine execution**: store/policy wiring verified with synthetic registries and a stub engine; real Qwen/GPU reconstruction not verified
<!-- release-sync:alternatives -->

<!-- release-sync:status -->
Public core status: — all declared sections verified
License: MIT License — Copyright (c) 2026 bluescv
<!-- release-sync:status -->
