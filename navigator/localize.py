"""Correlative scan matching: corrects the odometry pose by aligning the current lidar scan
with the occupancy grid built so far (likelihood field = distance to nearest mapped wall)."""
import math
import numpy as np
from scipy import ndimage


class ScanMatcher:
    """Aligns each lidar scan with the map built so far.

    Method : the map is turned into a distance field
    (distance to the nearest confirmed wall), read with bilinear interpolation so the result is
    smooth below the 2 cm cell size, and the x/y offset that minimises the squared distances of the
    scan end points is found with a coarse-to-fine search. The heading comes from the IMU, which
    does not drift in this simulation."""

    def __init__(self, grid, P):
        self.g, self.P = grid, P
        self.field = None
        self.field_t = -1.0
        self.wide_used = 0      # number of corrections found by the wide search (diagnostics)

    def refresh(self, t):
        if t - self.field_t < self.P["match_field_period"]:
            return
        occ = self.g.logodds > self.P["l_occ"] + 1.0          # only well-established walls
        if occ.sum() < 50:
            self.field = None
            return
        cap = self.P["match_cap"]
        if self.P["match_skeleton"]:
            # distance to the walls' centre lines minus half a board: zero on the board faces.
            # A wall band smeared to 10 cm by earlier pose errors then still pulls the scan to
            # its middle instead of letting it slide anywhere inside the band
            from skimage.morphology import skeletonize
            sk = skeletonize(occ)
            d = np.abs(ndimage.distance_transform_edt(~sk) * self.g.res - self.P["wall_half"])
            self.field = np.minimum(d, cap).astype(np.float64)
        else:
            self.field = np.minimum(ndimage.distance_transform_edt(~occ) * self.g.res, cap).astype(np.float64)
        gy, gx = np.gradient(self.field, self.g.res)
        self.gx, self.gy = gx, gy
        self.field_t = t

    def _sample(self, arr, ex, ey):
        g = self.g
        coords = np.vstack([ey / g.res + g.off - 0.5, ex / g.res + g.off - 0.5])
        return ndimage.map_coordinates(arr, coords, order=1, mode="nearest")

    def fit(self, ox, oy, a, rng, shifts):
        """mean squared (capped) distance of the scan end points to the mapped walls, for each
        candidate pose shift (dx, dy); None while there is no map yet"""
        if self.field is None:
            return None
        keep = rng < self.P["match_range"]
        if keep.sum() < 40:
            return None
        ex = (ox + rng * np.cos(a))[keep]; ey = (oy + rng * np.sin(a))[keep]
        cap = self.P["match_cap"]
        out = []
        for dx, dy in shifts:
            d = self._sample(self.field, ex + dx, ey + dy)
            out.append(float(np.mean(np.minimum(d, cap) ** 2)))
        return out

    def match(self, ox, oy, a, rng, odo_step=(0.0, 0.0)):
        """returns the (dx, dy) correction that best aligns the scan with the map, and the
        reduction of the mean squared end-point distance it achieves"""
        if self.field is None:
            return (0.0, 0.0), 0.0
        keep = rng < self.P["match_range"]
        if keep.sum() < 40:
            return (0.0, 0.0), 0.0
        ex = (ox + rng * np.cos(a))[keep]
        ey = (oy + rng * np.sin(a))[keep]
        cap = self.P["match_cap"]
        d0 = self._sample(self.field, ex, ey)
        use = d0 < 0.8 * cap                  # points far from any known wall belong to new areas
        if use.sum() < 30:
            return (0.0, 0.0), 0.0
        ex, ey = ex[use], ey[use]

        def cost(dx, dy):
            # candidates are evaluated together: (C,) offsets against (N,) points
            px = ex[None, :] + dx[:, None]
            py = ey[None, :] + dy[:, None]
            d = self._sample(self.field, px.ravel(), py.ravel()).reshape(px.shape)
            return np.mean(np.minimum(d, cap) ** 2, axis=1)

        best = np.zeros(2)
        c0 = float(cost(np.zeros(1), np.zeros(1))[0])
        cbest = c0
        # wheel slip: the wheels may have turned without the robot moving (wedged, pushing):
        # also start from 'only part / none of the odometry step happened'
        ox_, oy_ = odo_step
        if math.hypot(ox_, oy_) > 0.005:
            # only when the scan fits badly at the odometry pose, and clearly better without
            # (in a featureless corridor all hypotheses fit alike - then odometry is kept)
            ks = np.array([0.5, 1.0])
            cs = cost(-ks * ox_, -ks * oy_)
            j = int(np.argmin(cs))
            if cs[j] < self.P["slip_accept"] * cbest:
                cbest, best = float(cs[j]), np.array([-ks[j] * ox_, -ks[j] * oy_])
        centre = best.copy()
        # coarse-to-fine search (heading from the IMU is not searched)
        for half, step in ((0.04, 0.01), (0.01, 0.0025)):
            o = np.arange(-half, half + 1e-9, step)
            DX, DY = np.meshgrid(best[0] + o, best[1] + o, indexing="ij")
            c = cost(DX.ravel(), DY.ravel())
            k = int(np.argmin(c))
            if c[k] < cbest:
                cbest, best = float(c[k]), np.array([DX.ravel()[k], DY.ravel()[k]])
        if cbest > self.P["match_bad"]:
            # still a poor fit: the odometry may be far off (wheels slipping while wedged) -
            # search a wider window, accept only a clearly good alignment
            wb, wc = np.zeros(2), np.inf
            for half, step, centre in ((self.P["match_wide"], 0.02, None), (0.02, 0.005, 1), (0.005, 0.0025, 1)):
                c_ = wb if centre else np.zeros(2)
                o = np.arange(-half, half + 1e-9, step)
                DX, DY = np.meshgrid(c_[0] + o, c_[1] + o, indexing="ij")
                c = cost(DX.ravel(), DY.ravel())
                k = int(np.argmin(c))
                if c[k] < wc:
                    wc, wb = float(c[k]), np.array([DX.ravel()[k], DY.ravel()[k]])
            if wc < self.P["match_bad"] and wc < 0.5 * cbest:
                self.wide_used += 1
                return (float(wb[0]), float(wb[1])), c0 - wc
        if np.hypot(*(best - centre)) > self.P["match_max_shift"]:
            return (0.0, 0.0), 0.0
        return (float(best[0]), float(best[1])), c0 - cbest


class SlipCheck:
    """Map-independent check of the odometry: the current lidar scan is aligned with an earlier
    scan (key frame) instead of with the map. When the wheels spin while the robot is wedged
    (under a floating plate, against a corner) the odometry runs away; once the pose is off by
    more than the map matcher's capture range, the map matcher cannot pull it back. Comparing
    two raw scans does not depend on the map, so it also works when the map is already smeared.
    A correction is only applied when the scans clearly fit better with it (in a featureless
    corridor all shifts along the corridor fit alike - then the odometry is kept)."""

    def __init__(self, P, res=0.02, half=4.0):
        self.P, self.res, self.half = P, res, half
        self.ref = None          # (t, x, y, field, cx, cy)
        self.used = 0

    def _field(self, ex, ey, cx, cy):
        from scipy import ndimage
        n = int(2 * self.half / self.res)
        occ = np.zeros((n, n), bool)
        c = np.floor((ex - cx + self.half) / self.res).astype(int)
        r = np.floor((ey - cy + self.half) / self.res).astype(int)
        k = (c >= 0) & (c < n) & (r >= 0) & (r < n)
        occ[r[k], c[k]] = True
        return np.minimum(ndimage.distance_transform_edt(~occ) * self.res, self.P["slip_cap"])

    def _cost(self, field, cx, cy, ex, ey, dx, dy):
        from scipy import ndimage
        px = ex[None, :] + dx[:, None]; py = ey[None, :] + dy[:, None]
        coords = np.vstack([((py - cy + self.half) / self.res - 0.5).ravel(),
                            ((px - cx + self.half) / self.res - 0.5).ravel()])
        d = ndimage.map_coordinates(field, coords, order=1, mode="constant", cval=self.P["slip_cap"])
        return np.mean(np.minimum(d.reshape(px.shape), self.P["slip_cap"]) ** 2, axis=1)

    def check(self, t, pose, ox, oy, a, rng):
        """returns the (dx, dy) pose correction (0, 0 if none)"""
        P = self.P
        keep = rng < P["slip_range"]
        if keep.sum() < 60:
            self.ref = None
            return 0.0, 0.0
        ex = (ox + rng * np.cos(a))[keep]; ey = (oy + rng * np.sin(a))[keep]
        out = (0.0, 0.0)
        if self.ref is not None:
            t0, x0, y0, field, cx, cy = self.ref
            mx, my = pose[0] - x0, pose[1] - y0            # motion since the key frame (odometry)
            moved = math.hypot(mx, my)
            if moved > P["slip_min_move"]:
                sub = slice(None, None, max(1, len(ex) // 250))
                sx, sy = ex[sub], ey[sub]
                c_odo = float(self._cost(field, cx, cy, sx, sy, np.zeros(1), np.zeros(1))[0])
                # search shifts between 'all of the motion happened' (0) and 'none' (-m), and around
                h = moved + 0.04
                o = np.arange(-h, h + 1e-9, 0.02)
                DX, DY = np.meshgrid(o, o, indexing="ij"); DX, DY = DX.ravel(), DY.ravel()
                c = self._cost(field, cx, cy, sx, sy, DX, DY)
                k = int(np.argmin(c))
                o2 = np.arange(-0.02, 0.02 + 1e-9, 0.005)
                FX, FY = np.meshgrid(DX[k] + o2, DY[k] + o2, indexing="ij"); FX, FY = FX.ravel(), FY.ravel()
                c2 = self._cost(field, cx, cy, sx, sy, FX, FY)
                # among the near-best shifts take the one closest to the odometry (no correction)
                near = c2 <= c2.min() * 1.15 + 1e-6
                j = np.flatnonzero(near)[np.argmin(np.hypot(FX[near], FY[near]))]
                bx, by, cb = float(FX[j]), float(FY[j]), float(c2[j])
                # slip can only mean LESS motion than the wheels measured, in the same direction:
                # a shift that would make the robot move further, sideways or backwards is a false
                # fit (repeated wall patterns, e.g. a row of parallel boards), not slip
                cmx, cmy = mx + bx, my + by
                # checks run every few control steps, so real slip shows up as a series of small
                # corrections; one large jump at once is a scan matched to the wrong board
                plausible = (math.hypot(bx, by) <= P["slip_max_corr"]
                             and math.hypot(cmx, cmy) <= moved + P["slip_tol"]
                             and cmx * mx + cmy * my >= -P["slip_tol"] * moved
                             and math.hypot(bx, by) <= moved + P["slip_tol"])
                if (plausible and math.hypot(bx, by) > P["slip_min_corr"] and cb < P["slip_ratio"] * c_odo
                        and c_odo > P["slip_min_cost"]):
                    out = (bx, by)
                    self.used += 1
        # new key frame after enough real motion / time (taken from the corrected pose)
        if (self.ref is None or t - self.ref[0] > P["slip_keyframe_t"]
                or math.hypot(pose[0] + out[0] - self.ref[1], pose[1] + out[1] - self.ref[2]) > P["slip_keyframe_d"]):
            ex2, ey2 = ex + out[0], ey + out[1]
            cx, cy = pose[0] + out[0], pose[1] + out[1]
            self.ref = (t, cx, cy, self._field(ex2, ey2, cx, cy), cx, cy)
        return out


# Reference: the code in this file was generated with Claude (Anthropic LLM, claude.ai), Oct 2026,
# in a guided session for the Autonomous Robots Modularbeit (prompt: "implement navigation for maze 1").
