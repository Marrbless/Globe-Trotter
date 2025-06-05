import json
import xml.etree.ElementTree as ET
from world.world import World
from world.export import export_resources_json, export_resources_xml


def test_export_json(tmp_path):
    world = World(width=3, height=3)
    file = tmp_path / "res.json"
    export_resources_json(world, file)
    data = json.loads(file.read_text())
    assert isinstance(data, list)


def test_export_xml(tmp_path):
    world = World(width=3, height=3)
    file = tmp_path / "res.xml"
    export_resources_xml(world, file)
    tree = ET.parse(file)
    root = tree.getroot()
    assert root.tag == "resources"

