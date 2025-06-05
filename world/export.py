from __future__ import annotations

"""Utilities for exporting world data in various formats."""

from pathlib import Path
import json
import xml.etree.ElementTree as ET

from .world import World, Hex, ResourceType


def _hex_to_dict(hex_: Hex) -> dict:
    return {
        "coord": list(hex_.coord),
        "terrain": hex_.terrain,
        "resources": {r.value: amt for r, amt in hex_.resources.items()},
    }


def export_resources_json(world: World, path: str | Path) -> None:
    """Export hex resource data to a JSON file."""
    data = [_hex_to_dict(h) for h in world.all_hexes() if h.resources]
    with open(Path(path), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def export_resources_xml(world: World, path: str | Path) -> None:
    """Export hex resource data to an XML file."""
    root = ET.Element("resources")
    for hex_ in world.all_hexes():
        if not hex_.resources:
            continue
        hex_el = ET.SubElement(
            root,
            "hex",
            q=str(hex_.coord[0]),
            r=str(hex_.coord[1]),
            terrain=hex_.terrain,
        )
        for rtype, amt in hex_.resources.items():
            ET.SubElement(hex_el, "resource", type=rtype.value, amount=str(amt))
    tree = ET.ElementTree(root)
    tree.write(Path(path), encoding="utf-8", xml_declaration=True)


__all__ = ["export_resources_json", "export_resources_xml"]

