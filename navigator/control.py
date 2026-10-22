"""Pure-pursuit path follower with speed limits for a skid-steer base."""
import math
import numpy as np


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class Follower:
    def __init__(self, P):
        self.P = P
        self.path = None        # (N,2) world points
        self.idx = 0
        self.v = 0.0
        self.why = {}          # seconds spent per speed-limiting reason (diagnostics)

    def set_path(self, pts):
        self.path = np.asarray(pts, float) if pts is not None and len(pts) > 1 else None
        self.idx = 0

    def unverified_ahead(self, grid, dist, near):
        """True if the path between `near` and `dist` metres ahead crosses cells where the depth camera
        has not yet seen the camera-housing height band free (closer than `near` it cannot see)"""
        if self.path is None:
            return False
        seg = self.path[self.idx:]
        if len(seg) < 2:
            return False
        acc = np.r_[0, np.cumsum(np.hypot(*np.diff(seg, axis=0).T))]
        seg = seg[(acc > near) & (acc < dist)]
        if len(seg) == 0:
            return False
        r, c = grid.cell(seg[:, 0], seg[:, 1])
        ok = grid.inside(r, c)
        return bool((~grid.band_ok[r[ok], c[ok]]).any())

    def blocked_ahead(self, blocked_fn, dist):
        """True if any path point within `dist` ahead is now inside an obstacle's safety zone"""
        if self.path is None:
            return False
        seg = self.path[self.idx:]
        acc = np.r_[0, np.cumsum(np.hypot(*np.diff(seg, axis=0).T))] if len(seg) > 1 else np.zeros(len(seg))
        seg = seg[acc < dist]
        return blocked_fn(seg)

    def command(self, pose, dt, clearance, goal_dist, v_cap=None):
        """returns (v, w) in m/s, rad/s"""
        P = self.P
        if self.path is None:
            self.v = 0.0
            return 0.0, 0.0
        x, y, th = pose
        d = np.hypot(self.path[:, 0] - x, self.path[:, 1] - y)
        lo = self.idx
        hi = min(len(self.path), self.idx + 60)
        self.idx = lo + int(np.argmin(d[lo:hi]))
        # look-ahead point
        tgt = self.path[-1]
        acc = 0.0
        la = float(np.clip(1.6 * (clearance - P["inflate_obst"]) + P["lookahead_min"], P["lookahead_min"], P["lookahead"]))
        for i in range(self.idx, len(self.path) - 1):
            acc += np.hypot(*(self.path[i + 1] - self.path[i]))
            if acc >= la:
                tgt = self.path[i + 1]
                break
        alpha = wrap(math.atan2(tgt[1] - y, tgt[0] - x) - th)
        if abs(alpha) > P["turn_in_place_angle"]:
            self.why["turn_on_spot"] = self.why.get("turn_on_spot", 0) + dt
            self.v = 0.0
            return 0.0, float(np.clip(3.0 * alpha, -P["w_max"], P["w_max"]))
        # speed: limited by clearance, by distance to the end of the path, by heading error
        lim = {"clearance": P["v_max"] * float(np.clip((clearance - P["inflate_obst"]) / P["slow_clearance"], P["slow_min_factor"], 1.0)),
               "goal_brake": math.sqrt(2 * P["accel"] * max(goal_dist, 0.0)) + P["v_min"],
               "heading": P["v_max"] * max(0.25, math.cos(alpha) ** 2),
               "unverified": v_cap if v_cap is not None else 9.0,
               # narrow gap (only a few cm to spare on each side): creep through
               "tight": P["v_tight"] if clearance < P["tight_clearance"] else 9.0}
        k = min(lim, key=lim.get)
        v_t = lim[k]
        if v_t < 0.95 * P["v_max"]:
            self.why[k] = self.why.get(k, 0) + dt
        else:
            self.why["full_speed"] = self.why.get("full_speed", 0) + dt
        v_t = max(v_t, P["v_min"])
        self.v = min(v_t, self.v + P["accel"] * dt) if v_t > self.v else v_t
        L = max(np.hypot(tgt[0] - x, tgt[1] - y), 0.05)
        w = 2.0 * self.v * math.sin(alpha) / L + 1.0 * alpha
        return self.v, float(np.clip(w, -P["w_max"], P["w_max"]))


def wheel_speeds(v, w, P):
    r, B = P["wheel_radius"], P["track_eff"]
    wl = (v - w * B / 2) / r
    wr = (v + w * B / 2) / r
    s = max(abs(wl), abs(wr), 1e-9)
    if s > P["wheel_max"]:
        wl, wr = wl * P["wheel_max"] / s, wr * P["wheel_max"] / s
    return wl, wr


# Reference: the code in this file was generated with Claude (Anthropic LLM, claude.ai), Oct 2026,
# in a guided session for the Autonomous Robots Modularbeit (prompt: "implement navigation for maze 1").
