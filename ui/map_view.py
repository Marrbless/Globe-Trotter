import math
import pygame

HEX_SIZE = 30


def hex_to_pixel(q, r, size=HEX_SIZE):
    x = size * 3/2 * q
    y = size * math.sqrt(3) * (r + q / 2)
    return x, y


def pixel_to_hex(x, y, size=HEX_SIZE):
    q = (2/3 * x) / size
    r = (-1/3 * x + math.sqrt(3)/3 * y) / size
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
        pygame.init()
        self.world = world
        self.size = size
        self.screen = pygame.display.set_mode(size)
        self.camera = Camera(*size)
        self.clock = pygame.time.Clock()
        self.road_mode = False
        self.road_start = None

    def draw_hex(self, q, r, color, width=0):
        x, y = hex_to_pixel(q, r)
        x, y = self.camera.apply((x, y))
        corners = hex_corners(x, y)
        pygame.draw.polygon(self.screen, color, corners, width)

    def draw_roads(self):
        for road in getattr(self.world, "roads", []):
            x1, y1 = hex_to_pixel(*road.start)
            x2, y2 = hex_to_pixel(*road.end)
            x1, y1 = self.camera.apply((x1, y1))
            x2, y2 = self.camera.apply((x2, y2))
            pygame.draw.line(self.screen, (139, 69, 19), (x1, y1), (x2, y2), 4)

    def draw_map(self, selected=None):
        for r in range(self.world.height):
            for q in range(self.world.width):
                hex_data = self.world.hexes[r][q]
                terrain = hex_data["terrain"]
                color = terrain_color(terrain)
                self.draw_hex(q, r, color)
        self.draw_roads()
        if selected:
            q, r = selected
            self.draw_hex(q, r, (255, 255, 0), 3)
        if self.road_start:
            q, r = self.road_start
            self.draw_hex(q, r, (255, 165, 0), 3)

    def hex_at_pos(self, pos):
        x, y = self.camera.reverse(pos)
        q, r = pixel_to_hex(x, y)
        return self.world.get(q, r)

    def run(self):
        running = True
        selected = None
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        hex_data = self.hex_at_pos(event.pos)
                        if hex_data:
                            coords = (hex_data["q"], hex_data["r"])
                            if self.road_mode:
                                if self.road_start is None:
                                    self.road_start = coords
                                else:
                                    self.world.add_road(self.road_start, coords)
                                    self.road_start = None
                            else:
                                selected = coords
                    elif event.button == 4:
                        self.camera.change_zoom(0.1, event.pos)
                    elif event.button == 5:
                        self.camera.change_zoom(-0.1, event.pos)
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RETURN and selected:
                        return selected
                    elif event.key == pygame.K_r:
                        self.road_mode = not self.road_mode
                        self.road_start = None
            keys = pygame.key.get_pressed()
            pan_speed = 10
            if keys[pygame.K_LEFT]:
                self.camera.pan(pan_speed, 0)
            if keys[pygame.K_RIGHT]:
                self.camera.pan(-pan_speed, 0)
            if keys[pygame.K_UP]:
                self.camera.pan(0, pan_speed)
            if keys[pygame.K_DOWN]:
                self.camera.pan(0, -pan_speed)

            self.screen.fill((0, 0, 0))
            self.draw_map(selected)
            pygame.display.flip()
            self.clock.tick(60)
        return None


def terrain_color(name):
    mapping = {
        "plains": (110, 205, 88),
        "forest": (34, 139, 34),
        "mountains": (139, 137, 137),
        "hills": (107, 142, 35),
        "water": (65, 105, 225),
    }
    return mapping.get(name, (200, 200, 200))
