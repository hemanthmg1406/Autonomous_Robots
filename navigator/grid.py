"""Occupancy grid in a world-aligned frame whose origin is the start position."""
import math
import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize


class Grid:
    def __init__(self, P):
        self.P = P
        self.res = P["res"]
        self.n = int(round(P["size"] / self.res))
        self.off = self.n // 2                       # cell index of the start position
        self.logodds = np.zeros((self.n, self.n), np.float32)   # lidar evidence
        self.hz = np.zeros((self.n, self.n), bool)              # cells marked by a too-close slab sighting (see navigator)
        self.vis_count = np.zeros((self.n, self.n), np.uint8)   # floating/low obstacle evidence (depth camera, IR)
        self.green = np.zeros((self.n, self.n), bool)           # forbidden floor (rgb camera)
        self.green_ev = np.zeros((self.n, self.n), np.int8)     # green evidence: + green pixels, - plain floor
        self.seen = np.zeros((self.n, self.n), bool)            # observed at all
        self.cam_seen = np.zeros((self.n, self.n), bool)        # checked by the depth camera (floating walls)
        self.band_ok = np.zeros((self.n, self.n), bool)         # depth rays passed freely at camera-housing height
        self.swept = np.zeros((self.n, self.n), bool)           # covered by the robot body -> certainly free
        self.trail = np.zeros((self.n, self.n), bool)           # cells the robot centre has passed
        self.shadow = None
        self.recent = []                                        # (free cells, hit cells) of the last scans
        self.robot_xy = None
        self.shadow_ray = np.zeros((self.n, self.n), bool)      # just behind seen floating edges (camera rays)
        self.lidar_seen = np.zeros((self.n, self.n), bool)      # observed by the lidar (free or hit)
        self.looked = np.zeros((self.n, self.n), bool)          # lidar hits the camera has had in view, close enough

    # world <-> cell (row = y, col = x)
    def cell(self, x, y):
        return (np.floor(np.asarray(y) / self.res).astype(int) + self.off,
                np.floor(np.asarray(x) / self.res).astype(int) + self.off)

    def xy(self, r, c):
        return ((np.asarray(c) - self.off + 0.5) * self.res, (np.asarray(r) - self.off + 0.5) * self.res)

    def inside(self, r, c):
        return (r >= 0) & (r < self.n) & (c >= 0) & (c < self.n)

    def update_lidar(self, ox, oy, a, rng):
        """world-frame beams: origins (ox, oy), directions a, ranges rng (finite only)"""
        P = self.P
        ca, sa = np.cos(a), np.sin(a)
        # free space: samples along each ray, stopping a little before the hit
        step = self.res * 0.9
        nmax = int(P["lidar_max_use"] / step) + 1
        k = np.arange(nmax) * step
        free_len = np.minimum(rng, P["lidar_max_use"]) - P["free_stop"]  # don't clear the cells right before a hit
        fx = ox[:, None] + ca[:, None] * k[None, :]
        fy = oy[:, None] + sa[:, None] * k[None, :]
        m = k[None, :] < free_len[:, None]
        r, c = self.cell(fx[m], fy[m])
        ok = self.inside(r, c)
        r, c = r[ok], c[ok]
        lin = np.unique(r * self.n + c)
        # for the local refresh: free up to just before the hit (the band in front of a wall that
        # pose noise smeared is exactly what should be seen through)
        m2 = k[None, :] < (np.minimum(rng, P["local_radius"] + 0.5) - P["local_free_stop"])[:, None]
        r2, c2 = self.cell(fx[m2], fy[m2]); ok2 = self.inside(r2, c2)
        free_lin = np.unique(r2[ok2] * self.n + c2[ok2])
        cur = self.logodds.flat[lin]
        # confirmed walls are 'sticky': rays skimming a thin wall must not erase it
        self.logodds.flat[lin] = np.where(cur > P["l_sticky"], cur + P["sticky_clear"] * P["l_free"], cur + P["l_free"])
        self.seen.flat[lin] = True
        self.lidar_seen.flat[lin] = True
        # hits
        h = rng < P["lidar_max_use"]
        r, c = self.cell(ox[h] + ca[h] * rng[h], oy[h] + sa[h] * rng[h])
        ok = self.inside(r, c)
        lin = np.unique(r[ok] * self.n + c[ok])
        self.logodds.flat[lin] += P["l_hit"]
        self.seen.flat[lin] = True
        self.lidar_seen.flat[lin] = True
        np.clip(self.logodds, P["l_min"], P["l_max"], out=self.logodds)
        # the last few scans, taken from (nearly) the current pose, for the local refresh
        self.recent.append((free_lin, lin))
        if len(self.recent) > P["local_scans"]:
            self.recent.pop(0)

    def mark(self, layer, x, y):
        r, c = self.cell(x, y)
        ok = self.inside(r, c)
        layer[r[ok], c[ok]] = True
        self.seen[r[ok], c[ok]] = True

    def mark_open_rays(self, ox, oy, a):
        """beams without any return (nothing within the lidar's 12 m): the space along them is
        observed as empty for exploration purposes. The map's free/occupied evidence is not
        touched, because an object closer than the lidar's minimum range also returns 'inf'."""
        P = self.P
        step = self.res * 0.9
        k = np.arange(int(P["open_ray_from"] / step), int(P["lidar_max_use"] / step) + 1) * step
        x = ox[:, None] + np.cos(a)[:, None] * k[None, :]
        y = oy[:, None] + np.sin(a)[:, None] * k[None, :]
        r, c = self.cell(x.ravel(), y.ravel())
        ok = self.inside(r, c)
        self.lidar_seen[r[ok], c[ok]] = True

    def mark_shadow(self, x, y):
        r, c = self.cell(x, y)
        ok = self.inside(r, c)
        self.shadow_ray[r[ok], c[ok]] = True

    def mark_looked(self, x, y):
        """lidar hit points that were inside the camera's view and range: a pillar there would
        have been recognised"""
        r0, c0 = self.cell(x, y)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                r, c = r0 + dr, c0 + dc
                ok = self.inside(r, c)
                self.looked[r[ok], c[ok]] = True

    def pillar_candidates(self, known=()):
        """small round-ish lidar objects that the camera has not looked at yet: both pillars are
        cylinders (r = 0.1 m) that the lidar sees, while every wall is a long board"""
        P = self.P
        occ = (self.logodds > P["l_occ"]) & ~self.swept
        lab, n = ndimage.label(occ, structure=np.ones((3, 3)))
        if n == 0:
            return []
        out = []
        for i, sl in enumerate(ndimage.find_objects(lab)):
            ext = max(sl[0].stop - sl[0].start, sl[1].stop - sl[1].start) * self.res
            if ext > P["blob_max"]:
                continue
            m = lab[sl] == i + 1
            cnt = int(m.sum())
            if cnt < P["blob_min_cells"]:
                continue
            if self.looked[sl][m].mean() > 0.5:
                continue
            rr, cc = np.nonzero(m)
            # a pillar is an arc (the lidar sees its near half), a wall fragment is a straight strip
            ev = np.linalg.eigvalsh(np.cov(np.c_[cc, rr].T * self.res))
            minor, major = math.sqrt(max(ev[0], 0)), math.sqrt(max(ev[1], 1e-12))
            if minor < P["blob_min_width"] or minor / major < P["blob_min_ratio"]:
                continue
            x, y = self.xy(rr.mean() + sl[0].start, cc.mean() + sl[1].start)
            x, y = float(x), float(y)
            if any(math.hypot(x - kx, y - ky) < 0.4 for kx, ky in known):
                continue
            out.append((x, y, sl, m))
        return out

    def update_green(self, gx, gy, nx, ny):
        """green evidence: a cell counts as green once seen green; plain floor seen there later
        (several times) removes it again (projection errors of far pixels)"""
        P = self.P
        plus = np.zeros((self.n, self.n), bool)
        r, c = self.cell(gx, gy); ok = self.inside(r, c); plus[r[ok], c[ok]] = True
        r, c = self.cell(nx, ny); ok = self.inside(r, c)
        minus = np.zeros_like(plus); minus[r[ok], c[ok]] = True
        minus &= ~ndimage.binary_dilation(plus, iterations=1)
        ev = self.green_ev.astype(np.int16)
        ev[plus] += P["green_plus"]
        ev[minus] -= 1
        np.clip(ev, P["green_ev_min"], P["green_ev_max"], out=ev)
        self.green_ev[:] = ev
        new = self.green_ev >= P["green_plus"]
        fresh = new & ~self.green
        self.green = new
        self.seen |= plus | minus
        return fresh

    def add_obstacle(self, x, y, weight=1, skip_near_lidar=False):
        r, c = self.cell(x, y)
        ok = self.inside(r, c)
        if skip_near_lidar and ok.any():
            # the lidar maps normal walls precisely; camera points next to a lidar wall would only
            # make it thicker. Floating slabs and low boards are not lidar walls, so they stay.
            k = self.P["cam_lidar_gap"]
            near = np.zeros(ok.sum(), bool)
            rr, cc = r[ok], c[ok]
            for dr in range(-k, k + 1):
                for dc in range(-k, k + 1):
                    r2 = np.clip(rr + dr, 0, self.n - 1); c2 = np.clip(cc + dc, 0, self.n - 1)
                    near |= self.logodds[r2, c2] > self.P["l_occ"]
            idx = np.nonzero(ok)[0][~near]
            ok = np.zeros_like(ok); ok[idx] = True
        lin = np.unique(r[ok] * self.n + c[ok])
        self.vis_count.flat[lin] = np.minimum(self.vis_count.flat[lin].astype(int) + weight, 250)
        self.seen.flat[lin] = True
        self.cam_seen.flat[lin] = True

    @property
    def vis_obst(self):
        """camera obstacles: cells seen often enough AND part of a cluster (floating walls and low
        boards are long objects; isolated specks are depth/pose noise)"""
        m = self.vis_count >= self.P["vis_hits"]
        if not m.any():
            return m
        lab, n = ndimage.label(m, structure=np.ones((3, 3)))
        sizes = np.bincount(lab.ravel())
        keep = sizes >= self.P["vis_min_cluster"]
        keep[0] = False
        return keep[lab]

    def mark_free_points(self, x, y):
        """floor seen by the depth camera; each point also covers its neighbour cells, because
        far-away floor pixels are sparser than the grid"""
        r0, c0 = self.cell(x, y)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                r, c = r0 + dr, c0 + dc
                ok = self.inside(r, c)
                self.seen[r[ok], c[ok]] = True
                self.cam_seen[r[ok], c[ok]] = True

    def mark_band_clear(self, x, y):
        r0, c0 = self.cell(x, y)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                r, c = r0 + dr, c0 + dc
                ok = self.inside(r, c)
                self.band_ok[r[ok], c[ok]] = True

    def mark_robot(self, pose):
        """the space the robot body occupies right now is free (shrunk by a safety margin)"""
        x, y, th = pose
        self.robot_xy = (x, y)
        m = self.P["sweep_margin"]
        hx, hy = 0.10 - m, 0.116 - m
        k = np.arange(-1.0, 1.0001, self.res / max(hx, hy) * 0.7)
        u, v = np.meshgrid(k * hx, k * hy)
        c, s = np.cos(th), np.sin(th)
        r, cc = self.cell(x + c * u - s * v, y + s * u + c * v)
        ok = self.inside(r, cc)
        self.swept[r[ok], cc[ok]] = True
        self.band_ok[r[ok], cc[ok]] = True       # the robot's own body has been there
        rr, c0 = self.cell(x, y)
        self.trail[rr, c0] = True

    def occupied(self):
        # a lidar-height wall cannot be where the robot body has been; camera obstacles (floating
        # slabs above the low front of the robot, low boards) are never cleared this way
        return ((self.logodds > self.P["l_occ"]) & ~self.swept) | self.vis_obst

    def obstacle_distance(self, clear_disc=None, confident=False):
        """distance [m] from every cell to the nearest wall/obstacle and to the nearest green cell.

        Lidar walls: the mapped band is reduced to its centre line (skeleton) and half a board
        thickness is subtracted. Localisation errors of a few cm make a 5 cm board look 10-15 cm
        thick in the map; this keeps narrow passages open without trusting a thinner wall than real.
        clear_disc=(x, y, r): ignore obstacles inside this disc (the pillar we are driving to)"""
        lid = (self.logodds > (self.P["l_sticky"] if confident else self.P["l_occ"])) & ~self.swept
        if self.P["local_refresh"] and self.recent and self.robot_xy is not None:
            # near the robot the current scans are exact relative to the robot, while the map
            # has accumulated a few cm of pose noise (walls drawn too thick, gaps too narrow):
            # wall cells the latest rays passed through - away from any latest hit - are dropped
            R = int(self.P["local_radius"] / self.res)
            r0, c0 = self.cell(*self.robot_xy)
            sl = (slice(max(r0 - R, 0), r0 + R + 1), slice(max(c0 - R, 0), c0 + R + 1))
            fr = np.zeros((self.n, self.n), bool); hi = np.zeros((self.n, self.n), bool)
            for f_, h_ in self.recent:
                fr.flat[f_] = True; hi.flat[h_] = True
            keep_hit = ndimage.binary_dilation(hi[sl], iterations=1)
            drop = fr[sl] & ~keep_hit
            sub = lid[sl]
            sub &= ~drop
            lid[sl] = sub
        vis = self.vis_obst.copy()
        # cells right behind a seen floating edge (marked along the camera rays) count as obstacle
        # until the camera has seen the camera-housing height free there or the robot was there
        sh = self.shadow_ray & ~self.band_ok & ~self.swept
        if sh.any() and vis.any():
            # only behind accepted camera obstacles (clusters), not behind single noisy points
            k = int(round(self.P["shadow_dist"] / self.res)) + 1
            sh &= ndimage.binary_dilation(vis, iterations=k)
            lid_any = (self.logodds > self.P["l_occ"])
            # next to a lidar wall the camera just sees the wall: no shadow there
            off_wall = ndimage.distance_transform_edt(~lid_any) * self.res > self.P["shadow_wall_gap"]
            vis |= sh & off_wall
        if clear_disc is not None:
            x, y, r = clear_disc
            rr, cc = self.cell(x, y)
            k = int(r / self.res) + 1
            r0, r1, c0, c1 = max(rr - k, 0), min(rr + k + 1, self.n), max(cc - k, 0), min(cc + k + 1, self.n)
            yy, xx = np.mgrid[r0:r1, c0:c1]
            disc = np.hypot(yy - rr, xx - cc) * self.res <= r
            lid[r0:r1, c0:c1][disc] = False
            vis[r0:r1, c0:c1][disc] = False
        if confident and vis.any() and lid.any():
            # camera obstacles touching a lidar wall are that wall seen by the camera: the wall's
            # centre line already represents it
            near_wall = ndimage.distance_transform_edt(~lid) * self.res <= self.P["shadow_wall_gap"]
            vis &= ~near_wall
        # wall ends a few cm apart (boards meeting at a corner) are closed - the lidar sees
        # through such slits, but no robot fits; real passages are > 0.26 m wide
        k = int(round(self.P["wall_close_gap"] / 2 / self.res))
        if k > 0:
            yy, xx = np.mgrid[-k:k + 1, -k:k + 1]
            lid = ndimage.binary_closing(lid, structure=(yy ** 2 + xx ** 2) <= k * k)
        skel = skeletonize(lid)
        nb = ndimage.convolve(skel.astype(np.uint8), np.ones((3, 3), np.uint8), mode="constant")
        skel &= (nb >= 2) | (self.logodds > self.P["l_sticky"])      # single isolated cells are noise
        d_lid = ndimage.distance_transform_edt(~skel) * self.res - self.P["wall_half"]
        d_vis = ndimage.distance_transform_edt(~vis) * self.res
        d_obst = np.minimum(d_lid, d_vis)
        d_green = ndimage.distance_transform_edt(~self.green) * self.res
        return d_obst, d_green


    def add_hazard(self, x, y):
        """a too-close slab edge somewhere along these cells (distance not measurable)"""
        self.add_obstacle(x, y)
        r, c = self.cell(x, y); ok = self.inside(r, c)
        self.hz[r[ok], c[ok]] = True

    def clear_hazard(self, x, y):
        """the depth camera saw the band just below the robot's top free here: a hazard mark here
        was not the slab (its distance was only a guess)"""
        r0, c0 = self.cell(x, y)
        r = np.concatenate([r0 + dr for dr in (-1, 0, 1) for dc in (-1, 0, 1)])
        c = np.concatenate([c0 + dc for dr in (-1, 0, 1) for dc in (-1, 0, 1)])
        ok = self.inside(r, c)
        r, c = r[ok], c[ok]
        m = self.hz[r, c]
        self.vis_count[r[m], c[m]] = 0
        self.hz[r[m], c[m]] = False

# Reference: the code in this file was generated with Claude (Anthropic LLM, claude.ai), Oct 2026,
# in a guided session for the Autonomous Robots Modularbeit (prompt: "implement navigation for maze 1").
