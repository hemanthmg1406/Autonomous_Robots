"""Turns raw sensor data into robot-frame geometry: lidar beams, green floor, floating/low
obstacles and pillar observations."""
import math
import numpy as np
import cv2

# camera mounting in the robot frame (Rosbot.proto: camera slot (-0.027,0,0.165) + Astra offsets)
RGB_Y = 0.011
DEPTH_Y = 0.037
CAM_X = 0.0


class Perception:
    def __init__(self, P, lidar_n, cam_w, cam_h, cam_fov, dep_w, dep_h, dep_fov):
        self.P = P
        self.reject, self.seen_span, self.last_method = None, None, None
        i = np.arange(lidar_n)
        self.lidar_ang = math.pi - (i + 0.5) * 2 * math.pi / lidar_n   # calibrated against the map
        self.cw, self.ch = cam_w, cam_h
        self.f = (cam_w / 2) / math.tan(cam_fov / 2)
        self.cx, self.cy = cam_w / 2 - 0.5, cam_h / 2 - 0.5
        self.dw, self.dh = dep_w, dep_h
        self.fd = (dep_w / 2) / math.tan(dep_fov / 2)
        self.dcx, self.dcy = dep_w / 2 - 0.5, dep_h / 2 - 0.5
        # pixel grids for the (sub-sampled) depth image
        s = 4
        vv, uu = np.mgrid[0:dep_h:s, 0:dep_w:s]
        self.dsub = s
        self.du = (uu - self.dcx) / self.fd
        self.dv = (vv - self.dcy) / self.fd
        # rgb pixel grid (sub-sampled) for green ground projection
        s2 = 4
        vv, uu = np.mgrid[0:cam_h:s2, 0:cam_w:s2]
        self.csub = s2
        self.cu = (uu - self.cx) / self.f
        self.cv = (vv - self.cy) / self.f

    # ---------------------------------------------------------------- lidar
    def lidar(self, ranges):
        r = np.asarray(ranges, np.float64)
        ok = np.isfinite(r)
        return self.lidar_ang[ok], r[ok], r

    # ---------------------------------------------------------------- rgb
    def hsv(self, bgra):
        img = np.frombuffer(bgra, np.uint8).reshape(self.ch, self.cw, 4)[:, :, :3]
        return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    def green_points(self, hsv, depth_img=None):
        """robot-frame floor points: (green x, y), (not-green x, y).
        Where the depth camera sees the floor (>= 0.6 m) the distance comes from the depth image,
        which does not depend on the robot's pitch; closer, the flat-floor projection is used.
        Non-green floor points (depth agrees with the floor plane) are returned so that wrongly
        mapped green can be removed again."""
        s = self.csub
        h = hsv[::s, ::s]
        below = self.cv > 0.02                    # below the horizon only
        g = (h[:, :, 0] > 45) & (h[:, :, 0] < 75) & (h[:, :, 1] > 150) & (h[:, :, 2] > 60)
        x_flat = np.where(below, (self.P["cam_height"] - 0.01) / np.maximum(self.cv, 1e-3), np.inf)
        x = x_flat
        floor_ok = np.zeros_like(g)
        if depth_img is not None:
            z = depth_img[::s, ::s]
            if z.shape == x_flat.shape:
                zf = np.isfinite(z)
                # above the horizon both values are infinite (inf - inf = nan); those pixels are
                # excluded by 'below & zf' anyway, so the warning is silenced, results are unchanged
                with np.errstate(invalid="ignore"):
                    floor_ok = below & zf & (np.abs(z - x_flat) < 0.06 * x_flat + 0.03)
                    x = np.where(below & zf & (np.abs(z - x_flat) < 0.25 * x_flat), z, x_flat)
        keep = below & (x < self.P["green_max_dist"])
        mg = g & keep
        mn = ~g & floor_ok & keep
        y = -self.cu * x + RGB_Y
        return (x[mg] + CAM_X, y[mg]), (x[mn] + CAM_X, y[mn])

    def lidar_range_at(self, lidar_full, b_left, b_right):
        """median lidar range over the beams between two robot-frame bearings (left > right)"""
        n = len(lidar_full)
        lo = int(math.floor((math.pi - b_left) / (2 * math.pi) * n - 0.5))
        hi = int(math.ceil((math.pi - b_right) / (2 * math.pi) * n - 0.5))
        vals = [lidar_full[i % n] for i in range(lo, hi + 1)]
        vals = [v for v in vals if math.isfinite(v)]
        return (float(np.median(vals)), float(np.max(vals) - np.min(vals)), len(vals)) if vals else (None, None, 0)

    def pillar(self, hsv, color, depth_img, lidar_full):
        """Pillar observation in the robot frame: (x, y, dist_to_centre) or None.

        A sighting is only accepted when its distance is confirmed by two independent cues, because
        a pillar glimpsed through a gap between walls easily gets the distance of the wall beside
        the gap: depth camera (central blob columns) and lidar (beams inside the blob) must agree;
        when the pillar is too close for the depth camera (< 0.6 m), the lidar distance must agree
        with the apparent width of a cylinder of the known radius."""
        P = self.P
        R = P["pillar_radius"]
        if color == "blue":
            m = (hsv[:, :, 0] > 105) & (hsv[:, :, 0] < 135) & (hsv[:, :, 1] > 170) & (hsv[:, :, 2] > 40)
        else:
            m = (hsv[:, :, 0] > 20) & (hsv[:, :, 0] < 38) & (hsv[:, :, 1] > 170) & (hsv[:, :, 2] > 90)
        n, lab, stats, cent = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
        self.reject = None
        self.seen_span = None
        if n <= 1:
            return None
        k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        if stats[k, cv2.CC_STAT_AREA] < 40:
            return None
        x0, w = int(stats[k, cv2.CC_STAT_LEFT]), int(stats[k, cv2.CC_STAT_WIDTH])
        b_left = math.atan2(-(x0 - self.cx), self.f)
        b_right = math.atan2(-(x0 + w - 1 - self.cx), self.f)
        self.seen_span = (b_left, b_right)     # the colour was seen at these bearings (even if rejected)
        if x0 <= 2 or x0 + w >= self.cw - 2:
            # cut by the image border (typically very close up): position from the lidar alone,
            # if the lidar shows one compact surface inside the visible part
            return self.pillar_from_lidar(lidar_full, b_left, b_right, int(stats[k, cv2.CC_STAT_HEIGHT]))
        bearing = 0.5 * (b_left + b_right)
        half = 0.5 * (b_left - b_right)
        d_width = R / max(math.sin(half), 1e-3)                      # centre distance from apparent width
        # lidar: beams over the central 50 % of the blob
        q = 0.25 * (b_left - b_right)
        r_l, spread, nb = self.lidar_range_at(lidar_full, b_left - q, b_right + q)
        # most beams without a valid return (the pillar is inside the lidar's 0.2 m minimum
        # range): the few valid ones pass beside it and measure the wall behind - not usable
        n_span = max(1, int(round((b_left - b_right - 2 * q) / (2 * math.pi / len(lidar_full)))) + 1)
        if nb < 0.6 * n_span:
            r_l = None
        d_lidar = r_l + R + 0.02 * math.cos(bearing) if r_l is not None else None
        # depth: central 50 % of the blob's columns
        d_depth = None
        if depth_img is not None:
            c0, c1 = x0 + w // 4, x0 + w - w // 4
            ys, xs = np.nonzero(lab[:, c0:c1 + 1] == k)
            if len(xs) > 20:
                xs = xs + c0
                sel = np.random.default_rng(0).choice(len(xs), min(300, len(xs)), replace=False)
                dv = depth_img[ys[sel], xs[sel]]
                dv = dv[np.isfinite(dv)]
                if len(dv) > 15:
                    q25, q50, q75 = np.percentile(dv, [25, 50, 75])
                    if q75 - q25 > P["pillar_depth_iqr"]:
                        self.reject = f"depth inconsistent ({q25:.2f}-{q75:.2f} m)"
                        return None
                    d_depth = q50 / max(math.cos(bearing), 0.3) + R
        # two independent distance cues have to agree; a pillar partly hidden behind a wall edge
        # looks narrower than its distance implies, so the width cue exposes that case
        width_ok = lambda d: abs(d_width - d) <= P["pillar_width_tol"] * d
        dist = None
        # a sliver (pillar mostly hidden behind walls) gives an unreliable position: the distance
        # cues then usually measure the wall beside the gap
        sliver = d_width > P["pillar_sliver"] * min(d for d in (d_depth, d_lidar, d_width) if d is not None)
        if sliver and d_width > P["pillar_sliver"] * max(d for d in (d_depth, d_lidar, 0.0) if d is not None):
            self.reject = f"sliver (width distance {d_width:.2f})"
            return None
        if d_depth is not None and d_lidar is not None and abs(d_depth - d_lidar) <= P["pillar_agree"]:
            dist, self.last_method = d_lidar, "depth+lidar"
        elif d_depth is not None and width_ok(d_depth):
            dist, self.last_method = d_depth, "depth+width"
        elif d_depth is None and d_lidar is not None and width_ok(d_lidar):
            dist, self.last_method = d_lidar, "lidar+width"
        elif d_depth is None and d_lidar is None and d_width < P["near_width_max"]:
            # so close that both range sensors are blind (depth < 0.6 m, lidar < 0.2 m): the whole
            # pillar is in the image, so its width and height give the distance
            blob_h = int(stats[k, cv2.CC_STAT_HEIGHT])
            if blob_h >= 0.6 * min(self.f * P["pillar_min_height"] / max(d_width - R, 0.05), self.ch):
                dist, self.last_method = d_width, "near-width"
        if dist is None:
            self.reject = (f"depth {d_depth if d_depth is None else round(d_depth, 2)} / lidar "
                           f"{d_lidar if d_lidar is None else round(d_lidar, 2)} / width {d_width:.2f} disagree")
            return None
        return (CAM_X + dist * math.cos(bearing), RGB_Y + dist * math.sin(bearing), dist)

    # ---------------------------------------------------------------- depth
    def pillar_from_lidar(self, lidar_full, b_left, b_right, blob_h):
        P = self.P
        R = P["pillar_radius"]
        n = len(lidar_full)
        lo = int(math.floor((math.pi - b_left) / (2 * math.pi) * n - 0.5))
        hi = int(math.ceil((math.pi - b_right) / (2 * math.pi) * n - 0.5))
        idx = np.arange(lo, hi + 1) % n
        r = lidar_full[idx]
        ok = np.isfinite(r) & (r > 0.15)
        if ok.sum() < 3:
            self.reject = "image border, no lidar"
            return None
        r_min = float(np.min(r[ok]))
        if r_min + R > P["border_lidar_max"]:
            self.reject = "image border, too far for lidar-only"
            return None
        near = ok & (r < r_min + R)
        ang = self.lidar_ang[idx][near]
        # the near surface must look like a cylinder of radius R: angular size ~ 2 asin(R / d)
        d = r_min + R
        span = float(np.max(ang) - np.min(ang)) + 2 * math.pi / n
        expect = 2 * math.asin(min(0.99, R / d))
        edge = (ang.max() >= b_left - 1e-6) or (ang.min() <= b_right + 1e-6)
        if not edge and abs(span - expect) > 0.5 * expect:
            self.reject = f"image border, lidar arc {span:.2f} rad vs {expect:.2f} expected"
            return None
        if edge and span > 1.5 * expect:
            self.reject = "image border, lidar surface too wide"
            return None
        # the coloured blob must be as tall as a pillar this close (>= 0.3 m high) looks; a far
        # pillar glimpsed at the border next to a near wall end is much smaller
        expect_h = min(self.f * P["pillar_min_height"] / max(d - R, 0.05), self.ch)
        if blob_h < 0.6 * expect_h:
            self.reject = f"image border, blob {blob_h}px too small for {d:.2f} m"
            return None
        bearing = float(np.mean(ang))
        self.last_method = "lidar-arc"
        return (0.02 + d * math.cos(bearing), d * math.sin(bearing), d)

    def depth_array(self, buf):
        return np.ctypeslib.as_array(buf, (self.dh * self.dw,)).reshape(self.dh, self.dw)

    def band_clear_points(self, depth_img, lo=None, hi=None):
        """robot-frame (x, y) of places where a depth ray passed freely through the height band in
        which a floating wall would hit the robot's top (camera housing). Seeing the floor somewhere
        does not prove that the space above it is free - a floating slab leaves the floor visible."""
        P = self.P
        custom = lo is not None
        lo = P["band_lo"] if lo is None else lo
        hi = P["band_hi"] if hi is None else hi
        h0 = P["cam_height"]
        s = self.dsub
        z = depth_img[::s, ::s]
        dv, du = self.dv, self.du
        # only rows whose rays cross the band somewhere between the minimum range and band_max
        if custom:
            xs_ = np.arange(P["verify_near"], P["band_max"] + 1e-9, 0.05)
            hh = h0 - dv[:, 0][:, None] * xs_[None, :]
            rows = np.nonzero(((hh >= lo) & (hh <= hi)).any(axis=1))[0]
        else:
            rows = np.nonzero((dv[:, 0] > (h0 - hi) / P["band_max"] - 1e-9) & (dv[:, 0] < (h0 - lo) / P["verify_near"]))[0]
        if len(rows) == 0:
            return np.empty(0), np.empty(0)
        z, dv, du = z[rows], dv[rows], du[rows]
        zmax = np.where(np.isfinite(z), np.minimum(z, P["band_max"]), np.nan) - 0.05
        xs = np.arange(P["verify_near"], P["band_max"] + 1e-9, 0.05)
        X = xs[None, None, :]
        h = h0 - dv[..., None] * X
        ok = (h >= lo) & (h <= hi) & (X <= zmax[..., None])
        if not ok.any():
            return np.empty(0), np.empty(0)
        XX = np.broadcast_to(X, ok.shape)[ok]
        Y = (-du[..., None] * X + DEPTH_Y)
        Y = np.broadcast_to(Y, ok.shape)[ok]
        return XX + CAM_X, Y

    def obstacle_points(self, depth_img):
        """robot-frame (x, y) of points that are neither floor nor high enough to pass under"""
        s = self.dsub
        z = depth_img[::s, ::s]
        ok = np.isfinite(z) & (z < self.P["depth_max_use"])
        x = z[ok]
        y = -self.du[ok] * x + DEPTH_Y
        h = self.P["cam_height"] - self.dv[ok] * x
        m = (h > self.P["obst_z_min"]) & (h < self.P["obst_z_max"])
        floor = h <= self.P["obst_z_min"]
        # floating: an obstacle point above the lidar plane with nothing under it in the same
        # image column at about the same distance (a wall standing on the floor would continue
        # down there). Only such edges hide unknown slab area behind them.
        H = np.full(z.shape, np.nan); X = np.full(z.shape, np.nan)
        H[ok] = h; X[ok] = x
        cand = (H > self.P["float_z_min"]) & (H < self.P["obst_z_max"])
        low = H < self.P["float_z_min"] - 0.03
        floating = np.zeros(z.shape, bool)
        rows, cols = np.nonzero(cand)
        if len(rows):
            # for each candidate: any low point further down in its column at a similar range?
            Xl = np.where(low, X, np.nan)
            below = np.zeros(len(rows), bool)
            for k in range(1, z.shape[0]):
                rr = rows + k
                v = rr < z.shape[0]
                if not v.any():
                    break
                d = np.abs(Xl[np.minimum(rr, z.shape[0] - 1), cols] - X[rows, cols])
                below |= v & (d < self.P["float_same_range"])
            floating[rows[~below], cols[~below]] = True
        self.float_pts = (X[floating] + CAM_X, -self.du[floating] * X[floating] + DEPTH_Y)
        return (x[m] + CAM_X, y[m]), (x[floor] + CAM_X, y[floor])


    def near_hazard(self, hsv, depth_img):
        """bearings (as du = -tan(bearing)) of image columns showing a board closer than the depth
        camera's minimum range at the height of the camera housing. The depth camera returns no
        range closer than 0.6 m, and a floating slab above the lidar plane is invisible to the lidar:
        such a slab edge right in front of the robot appears only in the colour image (wood texture
        where the depth image has no value; far 'no value' areas show the white background)."""
        P = self.P
        s = self.dsub
        z = depth_img[::s, ::s]
        band = (self.dv >= P["hz_dv_lo"]) & (self.dv <= P["hz_dv_hi"])
        u = np.clip(np.round(self.cx + self.du * self.f).astype(int), 0, self.cw - 1)
        v = np.clip(np.round(self.cy + self.dv * self.f).astype(int), 0, self.ch - 1)
        H = hsv[v, u]
        wood = (H[..., 0] >= P["wood_h_lo"]) & (H[..., 0] <= P["wood_h_hi"]) & (H[..., 1] > P["wood_s_min"])
        hz = band & np.isinf(z) & wood
        cols = np.nonzero(hz.sum(axis=0) >= P["hz_min_rows"])[0]
        return self.du[0, cols]

# Reference: the code in this file was generated with Claude (Anthropic LLM, claude.ai), Oct 2026,
# in a guided session for the Autonomous Robots Modularbeit (prompt: "implement navigation for maze 1").
