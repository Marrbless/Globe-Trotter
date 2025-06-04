import dearpygui.dearpygui as dpg

class FactionCreationUI:
    """Faction creation window implemented with DearPyGui."""

    def __init__(self) -> None:
        dpg.create_context()
        self.races = ["Human", "Elf", "Dwarf", "Orc"]
        self.result = None

        dpg.create_viewport(title="Faction Creation", width=300, height=200)
        with dpg.window(label="Faction Creation", width=300, height=200, no_resize=True, no_move=True):
            dpg.add_input_text(label="Faction Name", tag="name")
            dpg.add_combo(self.races, label="Race", default_value=self.races[0], tag="race")
            dpg.add_color_edit(label="Color", default_value=(255, 255, 255), tag="color", no_alpha=True, display_hex=True)
            dpg.add_button(label="Confirm", callback=self._confirm)
        dpg.set_primary_window(dpg.last_container(), True)
        dpg.setup_dearpygui()
        dpg.show_viewport()

    def _confirm(self, sender, app_data):
        name = dpg.get_value("name")
        race = dpg.get_value("race")
        color = dpg.get_value("color")
        self.result = {
            "name": name,
            "race": race,
            "color": "#%02x%02x%02x" % (int(color[0]), int(color[1]), int(color[2])),
        }
        dpg.stop_dearpygui()

    def mainloop(self) -> None:
        while dpg.is_dearpygui_running():
            dpg.render_dearpygui_frame()
        dpg.destroy_context()

if __name__ == "__main__":
    ui = FactionCreationUI()
    ui.mainloop()
