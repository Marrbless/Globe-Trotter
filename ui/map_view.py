import math
import time
import dearpygui.dearpygui as dpg
from world.world import BIOME_COLORS
from game import settings

HEX_SIZE = 30


def hex_to_pixel(q, r, size=HEX_SIZE):
    x = size * 3 / 2 * q
    y = size * math.sqrt(3) * (r + q / 2)
    return x, y


def pixel_to_hex(x, y, size=HEX_SIZE):
    q = (2 / 3 * x) / size
    r = (-1 / 3 * x + math.sqrt(3) / 3 * y) / size
    return hex_round(q, r)


def hex_round(q, r):
    x = q
    z = r
    y = -x - z
    rx = round(x)
    ry = round(y)
    rz = round(z)

    x_diff = abs(rx - x)
    y_diff = abs(ry - y)
    z_diff = abs(rz - z)

    if x_diff > y_diff and x_diff > z_diff:
        rx = -ry - rz
    elif y_diff > z_diff:
        ry = -rx - rz
    else:
        rz = -rx - ry
    return int(rx), int(rz)


n_axis = 6
angles = [math.radians(60 * i) for i in range(n_axis)]


def hex_corners(x, y, size=HEX_SIZE):
    return [
        (x + size * math.cos(a), y + size * math.sin(a))
        for a in angles
    ]


class Camera:
    """Simple camera handling panning and zoom."""

    def __init__(self, width, height):
        self.offset_x = width // 2
        self.offset_y = height // 2
        self.zoom = 1.0

    def apply(self, pos):
        x, y = pos
        return (
            x * self.zoom + self.offset_x,
            y * self.zoom + self.offset_y,
        )

    def reverse(self, pos):
        x, y = pos
        return (
            (x - self.offset_x) / self.zoom,
            (y - self.offset_y) / self.zoom,
        )

    def pan(self, dx, dy):
        self.offset_x += dx
        self.offset_y += dy

    def change_zoom(self, delta, pivot):
        old = self.zoom
        self.zoom = max(0.2, min(4.0, self.zoom + delta))
        scale = self.zoom / old
        px, py = pivot
        self.offset_x = px - scale * (px - self.offset_x)
        self.offset_y = py - scale * (py - self.offset_y)


class MapView:
    def __init__(self, world, size=(800, 600), *, show_progress=False, total_ticks=None):
        self.world = world
        self.size = size
        self.show_progress = show_progress
        self.total_ticks = total_ticks or settings.TARGET_TICKS
        self.start_time = time.time() if show_progress else None
        self.camera = Camera(*size)
        self.road_mode = False
        self.road_start = None
        self.selected = None
        self.result = None
        self.layers = ["terrain", "elevation", "temperature", "rainfall"]
        self.layer_index = 0
        self.show_raw = False

        dpg.create_context()
        dpg.create_viewport(title="Map View", width=size[0], height=size[1])
        with dpg.window(tag="_map_window", width=size[0], height=size[1], no_move=True, no_resize=True, no_title_bar=True):
            if show_progress:
                self.progress = dpg.add_progress_bar(label="0%", width=size[0], default_value=0.0)
            self.canvas = dpg.add_drawlist(width=size[0], height=size[1], tag="_canvas")
        with dpg.window(tag="_layer_window", pos=(10, 10), width=120, height=135, no_resize=True, no_move=True, no_title_bar=True):
            dpg.add_text("Layers")
            dpg.add_button(label="Terrain (F1)", callback=self._select_layer, user_data=0)
            dpg.add_button(label="Elevation (F2)", callback=self._select_layer, user_data=1)
            dpg.add_button(label="Temperature (F3)", callback=self._select_layer, user_data=2)
            dpg.add_button(label="Rainfall (F4)", callback=self._select_layer, user_data=3)
            dpg.add_checkbox(label="Raw Data (B)", tag="_raw_mode", default_value=False, callback=self._toggle_raw)
        dpg.set_primary_window("_map_window", True)
        with dpg.handler_registry():
            dpg.add_mouse_click_handler(callback=self._on_click)
            dpg.add_mouse_drag_handler(button=dpg.mvMouseButton_Middle, callback=self._on_drag)
            dpg.add_mouse_wheel_handler(callback=self._on_scroll)
            dpg.add_key_press_handler(callback=self._on_key)
        dpg.setup_dearpygui()
        dpg.show_viewport()

    # event callbacks
    def _on_click(self, sender, app_data):
        if app_data == dpg.mvMouseButton_Left:
            mx, my = dpg.get_mouse_pos()
            hex_data = self.hex_at_pos((mx, my))
            if hex_data:
                coords = hex_data.coord
                if self.road_mode:
                    if self.road_start is None:
                        self.road_start = coords
                    else:
                        self.world.add_road(self.road_start, coords)
                        self.road_start = None
                else:
                    self.selected = coords

    def _on_drag(self, sender, app_data):
        dx, dy = app_data[1], app_data[2]
        self.camera.pan(dx, dy)

    def _on_scroll(self, sender, app_data):
        pos = dpg.get_mouse_pos()
        self.camera.change_zoom(app_data * 0.1, pos)

    def _on_key(self, sender, app_data):
        if app_data == dpg.mvKey_Return and self.selected:
            self.result = self.selected
            dpg.stop_dearpygui()
        elif app_data == dpg.mvKey_R:
            self.road_mode = not self.road_mode
            self.road_start = None
        elif app_data == dpg.mvKey_Tab:
            self.layer_index = (self.layer_index + 1) % len(self.layers)
        elif app_data == dpg.mvKey_F1:
            self.layer_index = 0
        elif app_data == dpg.mvKey_F2:
            self.layer_index = 1
        elif app_data == dpg.mvKey_F3:
            self.layer_index = 2
        elif app_data == dpg.mvKey_F4:
            self.layer_index = 3
        elif app_data == dpg.mvKey_B:
            self.show_raw = not self.show_raw
            dpg.set_value("_raw_mode", self.show_raw)

    def _select_layer(self, sender, app_data, user_data):
        """Callback from layer buttons to change the active layer."""
        self.layer_index = int(user_data)

    def _toggle_raw(self, sender, app_data):
        """Toggle between biome view and raw data layers."""
        self.show_raw = bool(app_data)

    def draw_hex(self, q, r, color, width=0):
        x, y = hex_to_pixel(q, r)
        x, y = self.camera.apply((x, y))
        corners = hex_corners(x, y)
        dpg.draw_polygon(corners, color=(0, 0, 0, 255), fill=color, thickness=width or 1, parent=self.canvas)

    def draw_roads(self):
        for road in getattr(self.world, "roads", []):
            x1, y1 = hex_to_pixel(*road.start)
            x2, y2 = hex_to_pixel(*road.end)
            x1, y1 = self.camera.apply((x1, y1))
            x2, y2 = self.camera.apply((x2, y2))
            dpg.draw_line((x1, y1), (x2, y2), color=(139, 69, 19, 255), thickness=4, parent=self.canvas)

    def draw_rivers(self):
        for seg in getattr(self.world, "rivers", []):
            x1, y1 = hex_to_pixel(*seg.start)
            x2, y2 = hex_to_pixel(*seg.end)
            x1, y1 = self.camera.apply((x1, y1))
            x2, y2 = self.camera.apply((x2, y2))
            dpg.draw_line((x1, y1), (x2, y2), color=(65, 105, 225, 255), thickness=3, parent=self.canvas)

    def draw_map(self):
        dpg.delete_item(self.canvas, children_only=True)
        tl = self.camera.reverse((0, 0))
        br = self.camera.reverse(self.size)
        qmin, rmin = pixel_to_hex(*tl)
        qmax, rmax = pixel_to_hex(*br)
        for r in range(rmin - 2, rmax + 3):
            for q in range(qmin - 2, qmax + 3):
                hex_data = self.world.get(q, r)
                if hex_data:
                    layer = self.layers[self.layer_index]
                    if not self.show_raw or layer == "terrain":
                        color = terrain_color(
                            "water" if hex_data.lake else hex_data.terrain
                        )
                    elif layer == "elevation":
                        color = grayscale_color(hex_data.elevation)
                    elif layer == "temperature":
                        color = grayscale_color(hex_data.temperature)
                    else:  # rainfall/moisture
                        color = grayscale_color(hex_data.moisture)
                    self.draw_hex(q, r, color, 0)
                    if self.show_raw and layer != "terrain":
                        x, y = hex_to_pixel(q, r)
                        x, y = self.camera.apply((x, y))
                        value = (
                            hex_data.elevation
                            if layer == "elevation"
                            else hex_data.temperature
                            if layer == "temperature"
                            else hex_data.moisture
                        )
                        dpg.draw_text(
                            (x - HEX_SIZE / 2, y - 6),
                            f"{value:.2f}",
                            color=(0, 0, 0, 255),
                            size=10,
                            parent=self.canvas,
                        )
        self.draw_roads()
        self.draw_rivers()
        if self.selected:
            q, r = self.selected
            self.draw_hex(q, r, (255, 255, 0, 255), 3)
        if self.road_start:
            q, r = self.road_start
            self.draw_hex(q, r, (255, 165, 0, 255), 3)

    def hex_at_pos(self, pos):
        x, y = self.camera.reverse(pos)
        q, r = pixel_to_hex(x, y)
        return self.world.get(q, r)

    def run(self):
        while dpg.is_dearpygui_running():
            self.draw_map()
            if self.show_progress:
                elapsed = time.time() - self.start_time
                total = self.total_ticks * settings.TICK_SECONDS
                fraction = min(elapsed / total, 1.0)
                dpg.set_value(self.progress, fraction)
                dpg.set_item_label(self.progress, f"{int(elapsed)}s / {int(total)}s")
            dpg.render_dearpygui_frame()
        dpg.destroy_context()
        return self.result


def worker_assignment_dialog(faction) -> int:
    """Simple slider window allowing manual worker assignment."""
    dpg.create_context()
    dpg.create_viewport(title="Assign Workers", width=250, height=100)
    with dpg.window(label="Assign Workers", width=250, height=100, no_resize=True, no_move=True):
        dpg.add_slider_int(
            label="Workers",
            tag="_workers",
            default_value=faction.workers.assigned,
            min_value=0,
            max_value=faction.citizens.count,
        )
        dpg.add_button(label="Confirm", callback=lambda s, a: dpg.stop_dearpygui())
    dpg.set_primary_window(dpg.last_container(), True)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    while dpg.is_dearpygui_running():
        dpg.render_dearpygui_frame()
    value = dpg.get_value("_workers")
    dpg.destroy_context()
    return int(value)


def terrain_color(name):
    return BIOME_COLORS.get(name, (200, 200, 200, 255))


def grayscale_color(value: float) -> tuple[int, int, int, int]:
    level = int(max(0.0, min(1.0, value)) * 255)
    return (level, level, level, 255)


if __name__ == "__main__":
    from world.world import World

    world = World(width=40, height=30)
    view = MapView(world)
    choice = view.run()
    if choice:
        print("Selected hex:", choice)
