"""Power-mode (nvpmodel) control through the jtop service, so no sudo is needed.

Modes that share the current mode's GPU/CPU power-gating masks switch live. The others
(7W on the Orin Nano Super, which gates GPU TPCs) need a reboot, and are only applied
when explicitly asked for, because jtop then reboots the board immediately.
"""
import time

from .telemetry import device_info

# Lowest budget first, so sweeps and charts read left to right.
BUDGET_ORDER = ["7W", "15W", "25W", "MAXN_SUPER", "MAXN"]


class RebootRequired(RuntimeError):
    pass


def budget_rank(name):
    return BUDGET_ORDER.index(name) if name in BUDGET_ORDER else len(BUDGET_ORDER)


def modes():
    """[{name, id, live, current}], lowest budget first. live = switchable without reboot."""
    from jtop import jtop

    with jtop() as j:
        nvp = j.nvpmodel
        out = [{"name": name, "id": i, "live": bool(nvp.status[i]), "current": i == nvp.id}
               for i, name in enumerate(nvp.models)]
    return sorted(out, key=lambda m: budget_rank(m["name"]))


def current():
    return device_info()["power_mode"]


def set_mode(name, *, reboot=False, settle_s=10.0, timeout_s=60.0, log=print):
    """Switch to `name` and wait until nvpmodel reports it. Returns the new device_info()."""
    from jtop import jtop

    with jtop() as j:
        nvp = j.nvpmodel
        if name not in nvp.models:
            raise ValueError(f"unknown power mode {name!r}; available: {', '.join(nvp.models)}")
        idx = nvp.models.index(name)
        if idx == nvp.id:
            return device_info()
        if not nvp.status[idx]:
            if not reboot:
                raise RebootRequired(f"{name} needs a reboot to apply (different power-gating masks)")
            log(f"  switching to {name}: the board reboots now")
            nvp.set_nvpmodel_id(idx, force=True)
            return None
        log(f"  switching power mode {nvp.models[nvp.id]} -> {name}")
        j.nvpmodel = idx

    t0 = time.monotonic()
    while current() != name:
        if time.monotonic() - t0 > timeout_s:
            raise RuntimeError(f"power mode did not change to {name} within {timeout_s:.0f}s")
        time.sleep(0.5)
    # Clocks and governors re-settle after the caps change.
    time.sleep(settle_s)
    return device_info()
