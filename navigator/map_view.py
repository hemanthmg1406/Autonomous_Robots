"""Live map window: draws the occupancy grid and the robot's plan while the controller runs.

Display only - nothing here feeds back into the navigation. Refreshed every P["map_view_period"]
seconds of simulation time; switched off with P["show_map"] = False or NAV_SHOW_MAP=0."""
import math
import os
import numpy as np

# BGR colours (OpenCV)
UNKNOWN = (128, 128, 128)
CAM_WALL = (40, 40, 220)            # floating / low obstacle seen by the depth camera
GREEN = (60, 170, 60)               # forbidden floor
INFLATED = (215, 205, 245)          # where the robot centre may not go (walls grown by the robot radius)
TRAIL = (0, 140, 255)               # cells the robot centre has passed
PATH = (200, 0, 200)
ROBOT = (0, 0, 0)
PILLAR = {"blue": (200, 90, 0), "yellow": (0, 210, 230)}
SIZE = 720                          # px, longest side of the window
MARGIN = 0.5                        # m around everything observed so far


class MapView:
    def __init__(self, grid, P, title):
        self.grid = grid
        self.P = P
        self.title = title
        self.next_t = 0.0
        self.cv2 = None
        self.error = None
        if not P["show_map"] or os.environ.get("NAV_SHOW_MAP", "") == "0":
            return
        try:
            import cv2
            cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
            self.cv2 = cv2
        except Exception as ex:          # e.g. opencv-python-headless has no windows
            self.error = f"{ex.__class__.__name__}: {ex}"

    def image(self, t, pose, path=None, pillars=None, goal=None, blocked=None):
        """BGR image of the map around everything observed so far (north = up)"""
        cv2, g = self.cv2, self.grid
        # crop: observed cells plus the robot, with a margin
        rows, cols = np.nonzero(g.seen)
        rr, cr = g.cell(pose[0], pose[1])
        m = int(MARGIN / g.res)
        r0 = max(min(rows.min() if len(rows) else rr, rr) - m, 0)
        r1 = min(max(rows.max() if len(rows) else rr, rr) + m + 1, g.n)
        c0 = max(min(cols.min() if len(cols) else cr, cr) - m, 0)
        c1 = min(max(cols.max() if len(cols) else cr, cr) + m + 1, g.n)
        sl = (slice(r0, r1), slice(c0, c1))
        # lidar layer as occupancy probability p = 1 / (1 + exp(-logodds)): white = free, black = wall
        p = 1.0 / (1.0 + np.exp(-g.logodds[sl]))
        shade = (255 * (1.0 - p)).astype(np.uint8)
        img = np.dstack([shade] * 3)
        img[~g.seen[sl]] = UNKNOWN
        if blocked is not None:
            img[blocked[sl] & ~(g.logodds[sl] > self.P["l_occ"])] = INFLATED
        img[g.green[sl]] = GREEN
        img[g.vis_obst[sl]] = CAM_WALL
        img[g.trail[sl]] = TRAIL
        img = img[::-1]                                   # row 0 of the grid is the lowest y
        s = SIZE / max(img.shape[:2])
        img = cv2.resize(img, (int(img.shape[1] * s), int(img.shape[0] * s)), interpolation=cv2.INTER_NEAREST)

        def px(x, y):                                     # world [m] -> window pixel
            return (int((x / g.res + g.off - c0) * s), int((r1 - (y / g.res + g.off)) * s))

        if path is not None:
            pts = np.array([px(x, y) for x, y in path], np.int32)
            cv2.polylines(img, [pts], False, PATH, 2, cv2.LINE_AA)
        for col, v in (pillars or {}).items():
            c = px(v[1], v[2])
            cv2.circle(img, c, max(int(0.1 / g.res * s), 3), PILLAR.get(col, (0, 0, 0)), -1, cv2.LINE_AA)
            cv2.circle(img, c, max(int(0.1 / g.res * s), 3), (0, 0, 0), 2 if col == goal else 1, cv2.LINE_AA)
        x, y, th = pose
        c = px(x, y)
        cv2.circle(img, c, max(int(0.116 / g.res * s), 3), ROBOT, 2, cv2.LINE_AA)
        cv2.line(img, c, px(x + 0.2 * math.cos(th), y + 0.2 * math.sin(th)), ROBOT, 2, cv2.LINE_AA)
        if img.shape[1] < SIZE:                           # keep the window (and the legend) full width
            pad = np.full((img.shape[0], SIZE - img.shape[1], 3), UNKNOWN, np.uint8)
            img = np.hstack([img, pad])
        # header and legend
        bar = np.full((52, img.shape[1], 3), 255, np.uint8)
        cv2.putText(bar, f"t = {t:6.2f} s   goal: {goal or 'done'}   grid {g.res * 100:.0f} cm",
                    (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        xk = 8
        for name, colour in (("unknown", UNKNOWN), ("lidar wall", (0, 0, 0)), ("camera obstacle", CAM_WALL),
                             ("green", GREEN), ("too close", INFLATED), ("trail", TRAIL), ("path", PATH)):
            cv2.rectangle(bar, (xk, 30), (xk + 12, 42), colour, -1)
            cv2.rectangle(bar, (xk, 30), (xk + 12, 42), (0, 0, 0), 1)
            cv2.putText(bar, name, (xk + 16, 41), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
            xk += 22 + int(cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0][0])
        return np.vstack([bar, img])

    def update(self, t, pose, path=None, pillars=None, goal=None, blocked=None):
        if self.cv2 is None or t < self.next_t:
            return
        self.next_t = t + self.P["map_view_period"]
        self.cv2.imshow(self.title, self.image(t, pose, path, pillars, goal, blocked))
        self.cv2.waitKey(1)                               # lets the window redraw


# Reference: the code in this file was generated with Claude (Anthropic LLM, Claude Code), Oct 2026,
# prompt: "visualize the occupancy grid when the controller runs". OpenCV drawing functions:
# OpenCV documentation (cv2.imshow, cv2.polylines, cv2.circle, cv2.putText).
