import tkinter as tk
from .faction_creation import FactionCreationUI  # for style consistency if needed
from game.buildings import ALL_BUILDINGS, Building


class DefenseBuildingUI(tk.Tk):
    """UI for selecting defensive structures."""

    def __init__(self):
        super().__init__()
        self.title("Choose Defensive Structures")
        self.vars = {}
        for i, b in enumerate(ALL_BUILDINGS):
            var = tk.BooleanVar(value=False)
            chk = tk.Checkbutton(self, text=b.name, variable=var)
            chk.grid(row=i, column=0, sticky="w", padx=5, pady=2)
            self.vars[b.name] = (var, b)
        tk.Button(self, text="Confirm", command=self.confirm).grid(
            row=len(ALL_BUILDINGS), column=0, pady=5
        )
        self.selected: list[Building] = []

    def confirm(self):
        self.selected = [b for name, (v, b) in self.vars.items() if v.get()]
        self.destroy()


def choose_defenses() -> list[Building]:
    ui = DefenseBuildingUI()
    ui.mainloop()
    return ui.selected
