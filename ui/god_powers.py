from __future__ import annotations

"""Simple UI for invoking god powers using DearPyGui."""

from typing import Dict
import dearpygui.dearpygui as dpg

from game.game import Game
from game.god_powers import GodPower


class GodPowersUI:
    def __init__(self, game: Game) -> None:
        self.game = game
        dpg.create_context()
        dpg.create_viewport(title="God Powers", width=220, height=150)
        with dpg.window(label="God Powers", width=220, height=150, no_resize=True, no_move=True):
            self.container = dpg.add_group()
        dpg.set_primary_window(dpg.last_container(), True)
        dpg.setup_dearpygui()
        dpg.show_viewport()
        self.buttons: Dict[str, int] = {}

    def _refresh(self) -> None:
        for power in self.game.god_powers.values():
            label = f"{power.name}"
            if power.name not in self.buttons:
                dpg.push_container_stack(self.container)
                self.buttons[power.name] = dpg.add_button(
                    label=label,
                    callback=self._make_callback(power),
                )
                dpg.add_tooltip(self.buttons[power.name], label=power.description)
                dpg.pop_container_stack()
            # update enabled state
            enabled = (
                power in self.game.available_powers()
                and self.game.power_cooldowns.get(power.name, 0) <= 0
            )
            dpg.configure_item(self.buttons[power.name], enabled=enabled)
            if enabled:
                dpg.set_item_label(
                    self.buttons[power.name],
                    f"{power.name} ({power.cooldown}t cd)"
                )

    def _make_callback(self, power: GodPower):
        def cb(sender=None, app_data=None):
            try:
                self.game.use_power(power.name)
            except ValueError as exc:
                print(exc)
        return cb

    def mainloop(self) -> None:
        while dpg.is_dearpygui_running():
            self._refresh()
            dpg.render_dearpygui_frame()
        dpg.destroy_context()


def launch(game: Game) -> None:
    ui = GodPowersUI(game)
    ui.mainloop()
