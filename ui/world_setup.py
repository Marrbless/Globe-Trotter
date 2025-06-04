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
        with dpg.window(label="World Setup", pos=(10, 10), width=250, height=270):
            dpg.add_slider_int(
                label="Seed",
                tag="seed",
                min_value=0,
                max_value=99999,
                default_value=self.settings.seed,
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
                label="Tectonic Activity",
                tag="tectonic",
                min_value=0.0,
                max_value=1.0,
                default_value=self.settings.plate_activity,
                callback=self._update_world,
            )
            dpg.add_button(label="Confirm", callback=self._confirm)

    def _update_world(self, sender, app_data):
        """Regenerate world when any slider changes."""

        self.settings.seed = dpg.get_value("seed")
        self.settings.sea_level = dpg.get_value("sea_level")
        self.settings.temperature = dpg.get_value("temperature")
        self.settings.moisture = dpg.get_value("moisture")
        self.settings.plate_activity = dpg.get_value("tectonic")

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
