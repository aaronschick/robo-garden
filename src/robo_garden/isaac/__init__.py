"""Isaac Sim bridge package.

Usage:
    from robo_garden.isaac import get_bridge

    bridge = get_bridge()
    bridge.connect("ws://localhost:8765")  # no-op if Isaac Sim not running
    bridge.send_robot("my_robot", path_to_mjcf)
"""

from robo_garden.isaac.bridge import IsaacBridge

_bridge: IsaacBridge | None = None


def get_bridge() -> IsaacBridge:
    """Return the module-level IsaacBridge singleton (creates it on first call)."""
    global _bridge
    if _bridge is None:
        _bridge = IsaacBridge()
    return _bridge
