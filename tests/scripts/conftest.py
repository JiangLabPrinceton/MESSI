"""Test-process isolation for the V1 entry driver's NMP firewall.

``run_rf_fusion_v1_entry.assert_no_nmp`` scans ``sys.modules`` for any NetMHCIIpan module, because
in production the entry runtime is one process and anything in ``sys.modules`` really was imported
by it -- NMP is post-hoc only and must never reach the entry runtime.

Under pytest the whole repository shares ONE interpreter, so an unrelated suite (``tests/epitope_head``
legitimately imports ``epitope_head.data.netmhciipan_mutation``) leaves that module resident and
every subsequent driver test trips a guard about code it never ran. That is a false positive about
the test harness, not a finding about the driver.

This fixture hides ONLY modules that were already resident before the test began, and restores them
afterwards. A module the driver imports DURING the test is still resident and still trips the guard,
so the firewall keeps its teeth -- which is the whole point of not simply relaxing it.
"""

from __future__ import annotations

import sys

import pytest


def _nmp_modules() -> list[str]:
    return [m for m in sys.modules if "netmhciipan" in m.lower() or "netmhc_ii" in m.lower()]


@pytest.fixture(autouse=True)
def _hide_preimported_nmp_modules():
    stashed = {name: sys.modules.pop(name) for name in _nmp_modules()}
    try:
        yield
    finally:
        sys.modules.update(stashed)
