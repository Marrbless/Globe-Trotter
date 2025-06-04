import dearpygui.dearpygui as dpg
from game.buildings import ALL_DEFENSIVE_BUILDINGS, Building

class DefenseBuildingUI:
    """Selection window for defensive structures using DearPyGui."""

    def __init__(self) -> None:
        dpg.create_context()
        dpg.create_viewport(title="Choose Defensive Structures", width=250, height=200)
        with dpg.window(label="Choose Defensive Structures", width=250, height=200, no_resize=True, no_move=True):
            self._checks = {}
            for b in ALL_DEFENSIVE_BUILDINGS:
                self._checks[b.name] = dpg.add_checkbox(label=b.name, default_value=False)
            dpg.add_button(label="Confirm", callback=self._confirm)
        dpg.set_primary_window(dpg.last_container(), True)
        dpg.setup_dearpygui()
        dpg.show_viewport()
        self.selected: list[Building] = []

    def _confirm(self, sender, app_data):
        for b in ALL_DEFENSIVE_BUILDINGS:
            if dpg.get_value(self._checks[b.name]):
                self.selected.append(b)
        dpg.stop_dearpygui()

    def mainloop(self) -> None:
        while dpg.is_dearpygui_running():
            dpg.render_dearpygui_frame()
        dpg.destroy_context()


def choose_defenses() -> list[Building]:
    ui = DefenseBuildingUI()
    ui.mainloop()
    return ui.selected
