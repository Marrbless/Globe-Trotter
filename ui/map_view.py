import math
import dearpygui.dearpygui as dpg
from world.generation import BIOME_COLORS

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
    def __init__(self, world, size=(800, 600)):
        self.world = world
        self.size = size
        self.camera = Camera(*size)
        self.road_mode = False
        self.road_start = None
        self.selected = None
        self.result = None

        dpg.create_context()
        dpg.create_viewport(title="Map View", width=size[0], height=size[1])
        with dpg.window(tag="_map_window", width=size[0], height=size[1], no_move=True, no_resize=True, no_title_bar=True):
            self.canvas = dpg.add_drawlist(width=size[0], height=size[1], tag="_canvas")
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
                    color = terrain_color("water" if hex_data.lake else hex_data.terrain)
                    self.draw_hex(q, r, color, 0)
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
            dpg.render_dearpygui_frame()
        dpg.destroy_context()
        return self.result


def terrain_color(name):
    return BIOME_COLORS.get(name, (200, 200, 200, 255))
