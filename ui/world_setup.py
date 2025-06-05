"""World setup interface with real-time map preview."""

from __future__ import annotations

import dearpygui.dearpygui as dpg

from world.world import WorldSettings, World
from ui.map_view import MapView


class WorldSetupUI:
    """UI allowing the player to tweak world generation settings."""

    def __init__(self) -> None:
        # Use MapView to initialize DearPyGui context and viewport
        self.settings = WorldSettings()
        self.world = World(
            width=self.settings.width,
            height=self.settings.height,
            settings=self.settings,
        )
        self.view = MapView(self.world, size=(800, 600))
        self.result: WorldSettings | None = None

        # Sliders window layered on top of the MapView viewport
        with dpg.window(label="World Setup", pos=(10, 10), width=250, height=450):
            dpg.add_slider_int(
                label="Seed",
                tag="seed",
                min_value=0,
                max_value=99999,
                default_value=self.settings.seed,
                callback=self._update_world,
            )
            dpg.add_slider_int(
                label="Width",
                tag="width",
                min_value=10,
                max_value=200,
                default_value=self.settings.width,
                callback=self._update_world,
            )
            dpg.add_slider_int(
                label="Height",
                tag="height",
                min_value=10,
                max_value=200,
                default_value=self.settings.height,
                callback=self._update_world,
            )
            dpg.add_slider_float(
                label="Sea Level",
                tag="sea_level",
                min_value=0.0,
                max_value=1.0,
                default_value=self.settings.sea_level,
                callback=self._update_world,
            )
            dpg.add_slider_float(
                label="Temperature",
                tag="temperature",
                min_value=0.0,
                max_value=1.0,
                default_value=self.settings.temperature,
                callback=self._update_world,
            )
            dpg.add_slider_float(
                label="Rainfall",
                tag="moisture",
                min_value=0.0,
                max_value=1.0,
                default_value=self.settings.moisture,
                callback=self._update_world,
            )
            dpg.add_slider_float(
                label="Mountain Elev",
                tag="mountain_elev",
                min_value=0.0,
                max_value=1.0,
                default_value=self.settings.mountain_elev,
                callback=self._update_world,
            )
            dpg.add_slider_float(
                label="Hill Elev",
                tag="hill_elev",
                min_value=0.0,
                max_value=1.0,
                default_value=self.settings.hill_elev,
                callback=self._update_world,
            )
            dpg.add_slider_float(
                label="Tundra Temp",
                tag="tundra_temp",
                min_value=0.0,
                max_value=1.0,
                default_value=self.settings.tundra_temp,
                callback=self._update_world,
            )
            dpg.add_slider_float(
                label="Desert Rain",
                tag="desert_rain",
                min_value=0.0,
                max_value=1.0,
                default_value=self.settings.desert_rain,
                callback=self._update_world,
            )
            dpg.add_slider_float(
                label="Tectonic Activity",
                tag="tectonic",
                min_value=0.0,
                max_value=1.0,
                default_value=self.settings.plate_activity,
                callback=self._update_world,
            )
            dpg.add_slider_float(
                label="Rainfall Intensity",
                tag="rainfall_intensity",
                min_value=0.0,
                max_value=1.0,
                default_value=self.settings.rainfall_intensity,
                callback=self._update_world,
            )
            dpg.add_slider_float(
                label="Disaster Intensity",
                tag="disaster_intensity",
                min_value=0.0,
                max_value=1.0,
                default_value=self.settings.disaster_intensity,
                callback=self._update_world,
            )
            dpg.add_slider_float(
                label="Base Height",
                tag="base_height",
                min_value=0.0,
                max_value=1.0,
                default_value=self.settings.base_height,
                callback=self._update_world,
            )
            dpg.add_checkbox(
                label="World Changes",
                tag="world_changes",
                default_value=self.settings.world_changes,
                callback=self._update_world,
            )
            dpg.add_button(label="Confirm", callback=self._confirm)

    def _update_world(self, sender, app_data):
        """Regenerate world when any slider changes."""

        self.settings.seed = dpg.get_value("seed")
        self.settings.width = dpg.get_value("width")
        self.settings.height = dpg.get_value("height")
        self.settings.sea_level = dpg.get_value("sea_level")
        self.settings.temperature = dpg.get_value("temperature")
        self.settings.moisture = dpg.get_value("moisture")
        self.settings.mountain_elev = dpg.get_value("mountain_elev")
        self.settings.hill_elev = dpg.get_value("hill_elev")
        self.settings.tundra_temp = dpg.get_value("tundra_temp")
        self.settings.desert_rain = dpg.get_value("desert_rain")
        self.settings.plate_activity = dpg.get_value("tectonic")
        self.settings.rainfall_intensity = max(
            0.0, min(1.0, dpg.get_value("rainfall_intensity"))
        )
        self.settings.disaster_intensity = max(
            0.0, min(1.0, dpg.get_value("disaster_intensity"))
        )
        self.settings.base_height = max(
            0.0, min(1.0, dpg.get_value("base_height"))
        )
        self.settings.world_changes = dpg.get_value("world_changes")

        self.world = World(
            width=self.settings.width,
            height=self.settings.height,
            settings=self.settings,
        )
        self.view.world = self.world

    def _confirm(self, sender, app_data):
        self.result = self.settings
        dpg.stop_dearpygui()

    def mainloop(self) -> WorldSettings | None:
        while dpg.is_dearpygui_running():
            self.view.draw_map()
            dpg.render_dearpygui_frame()
        dpg.destroy_context()
        return self.result


def create_world() -> WorldSettings | None:
    ui = WorldSetupUI()
    return ui.mainloop()


if __name__ == "__main__":
    settings = create_world()
    if settings:
        print(settings)
