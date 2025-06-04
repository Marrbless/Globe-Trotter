import tkinter as tk
from tkinter import colorchooser
from tkinter import ttk

class FactionCreationUI(tk.Tk):
    """Simple GUI for creating a faction."""

    def __init__(self):
        super().__init__()
        self.title("Faction Creation")
        self.resizable(False, False)

        # Faction name input
        tk.Label(self, text="Faction Name:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.name_var = tk.StringVar()
        tk.Entry(self, textvariable=self.name_var, width=20).grid(row=0, column=1, padx=5, pady=5)

        # Race selection
        tk.Label(self, text="Race:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.races = ["Human", "Elf", "Dwarf", "Orc"]
        self.race_var = tk.StringVar(value=self.races[0])
        ttk.OptionMenu(self, self.race_var, self.races[0], *self.races).grid(row=1, column=1, padx=5, pady=5, sticky="w")

        # Color picker
        tk.Label(self, text="Color:").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.color_var = "#ffffff"
        self.color_button = tk.Button(self, bg=self.color_var, width=4, command=self.choose_color)
        self.color_button.grid(row=2, column=1, padx=5, pady=5, sticky="w")

        # Confirm button
        tk.Button(self, text="Confirm", command=self.confirm).grid(row=3, column=0, columnspan=2, pady=10)

    def choose_color(self):
        color = colorchooser.askcolor(color=self.color_var)[1]
        if color:
            self.color_var = color
            self.color_button.configure(bg=self.color_var)

    def confirm(self):
        print("Faction Name:", self.name_var.get())
        print("Race:", self.race_var.get())
        print("Color:", self.color_var)
        self.destroy()


if __name__ == "__main__":
    app = FactionCreationUI()
    app.mainloop()
