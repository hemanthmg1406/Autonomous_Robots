"""Grid planner: inflated cost map + minimum-cost path (scikit-image MCP, Dijkstra-like)."""
import numpy as np
from scipy import ndimage
from skimage.graph import MCP_Geometric


class Planner:
    def __init__(self, grid, P):
        self.g, self.P = grid, P
        self.d_obst = None
        self.blob_targets = None     # bool mask of viewpoints for pillar candidates (set by the navigator)
        self.last_tier = None
        self.heading = None          # robot heading [rad], set by the navigator before exploring

    def build(self, clear_disc=None, inflate=None, confident=False):
        """cost map; with an inflation below the normal one, the cells that are only free thanks
        to the smaller margin are remembered in self.narrow (penalised when choosing frontiers)"""
        g, P = self.g, self.P
        infl = P["inflate_obst"] if inflate is None else inflate
        d_obst, d_green = g.obstacle_distance(clear_disc, confident)
        self.d_obst, self.d_green = d_obst, d_green
        blocked = (d_obst < infl) | (d_green < P["inflate_green"])
        soft = np.clip((P["soft_dist"] - d_obst) / (P["soft_dist"] - P["inflate_obst"]), 0, 1)
        cost = 1.0 + P["soft_cost"] * soft ** 2
        cost[~g.seen] *= P["unknown_cost"]
        # elsewhere, where the depth camera has not yet seen the camera-housing height free, a
        # floating slab may hide (only its near edge is ever seen): prefer verified ground
        cost[~g.band_ok] *= P["unverified_cost"]
        # where the robot centre has already been is always passable: the way back can never close
        trail = ndimage.binary_dilation(g.trail, iterations=1)
        cost[blocked & trail] = 1.0 + P["soft_cost"] + P["trail_cost"]
        blocked = blocked & ~trail
        self.narrow = ~blocked & (d_obst < P["inflate_obst"])
        if clear_disc is not None:
            # the robot centre itself may not enter the cleared disc: obstacles erased there could
            # otherwise open a 'hole' through a wall standing right next to the pillar
            x, y, r = clear_disc
            rr, cc = g.cell(x, y)
            k = int(r / g.res) + 1
            r0, r1, c0, c1 = max(rr - k, 0), min(rr + k + 1, g.n), max(cc - k, 0), min(cc + k + 1, g.n)
            yy, xx = np.mgrid[r0:r1, c0:c1]
            blocked[r0:r1, c0:c1] |= np.hypot(yy - rr, xx - cc) * g.res <= r
            self.narrow &= ~blocked
        cost[blocked] = np.inf
        self.blocked, self.cost = blocked, cost

    def _window(self, rc_list, pad=40):
        g = self.g
        rows, cols = np.nonzero(g.seen)
        rr = np.concatenate([rows, [r for r, c in rc_list]])
        cc = np.concatenate([cols, [c for r, c in rc_list]])
        r0, r1 = max(int(rr.min()) - pad, 0), min(int(rr.max()) + pad + 1, g.n)
        c0, c1 = max(int(cc.min()) - pad, 0), min(int(cc.max()) + pad + 1, g.n)
        return r0, r1, c0, c1

    def free_start(self, rc, max_cells=14):
        """robot cell, or the nearest non-blocked cell if the robot is inside an inflated zone"""
        r, c = rc
        if not self.blocked[r, c]:
            return rc
        r0, r1 = max(r - max_cells, 0), min(r + max_cells + 1, self.g.n)
        c0, c1 = max(c - max_cells, 0), min(c + max_cells + 1, self.g.n)
        sub = ~self.blocked[r0:r1, c0:c1]
        if not sub.any():
            return None
        ys, xs = np.nonzero(sub)
        k = np.argmin((ys + r0 - r) ** 2 + (xs + c0 - c) ** 2)
        return (int(ys[k] + r0), int(xs[k] + c0))

    def plan(self, start_rc, goal_mask=None, frontier=False, prefer=None):
        """Returns list of (row, col) or None. goal_mask: bool array of acceptable end cells."""
        s = self.free_start(start_rc)
        if s is None:
            return None
        extra = [s]
        if goal_mask is not None:
            gr, gc = np.nonzero(goal_mask & ~self.blocked)
            if len(gr) == 0:
                return None
            extra += [(int(gr.min()), int(gc.min())), (int(gr.max()), int(gc.max()))]
        r0, r1, c0, c1 = self._window(extra)
        cost = self.cost[r0:r1, c0:c1]
        m = MCP_Geometric(cost, fully_connected=True)
        if frontier:
            # choose the frontier by (nearly) plain path length: the wall-distance cost that keeps
            # the robot centred would otherwise make every corridor look many times longer than
            # open floor; the route itself is planned with the full cost afterwards
            sel = np.where(np.isfinite(cost), 1.0 + self.P["explore_soft_share"] * (cost - 1.0), np.inf)
            sel = sel + self.P["narrow_penalty"] * self.narrow[r0:r1, c0:c1]
            msel = MCP_Geometric(sel, fully_connected=True)
            cum, _ = msel.find_costs([(s[0] - r0, s[1] - c0)])
            rr, cc = np.mgrid[r0:r1, c0:c1]
            dist = np.hypot(rr - s[0], cc - s[1]) * self.g.res
            cand = None
            self.last_tier = None
            tiers = self.frontier_mask()
            has_blob = self.blob_targets is not None and self.blob_targets.any()
            blob = tiers[0] if has_blob else np.zeros_like(tiers[0])
            if has_blob:
                tiers = tiers[1:]
            # pillar-candidate viewpoints compete with the frontiers (with a bonus) instead of
            # always coming first: a far, doubtful candidate must not pull the robot away from
            # the corridor it is exploring
            tiers = [tiers[0] | tiers[1] | blob] + tiers[2:]
            names = [2, 3]
            for tier, fm in enumerate(tiers):
                fm = fm[r0:r1, c0:c1]
                self.last_tier = names[min(tier, 1)]
                reach = fm & np.isfinite(cum)
                cand = reach & (dist > self.P["frontier_min_dist"])
                if not cand.any():
                    cand = reach & (dist > 2 * self.g.res)  # only near targets left: take those
                if cand.any():
                    break
            if not cand.any():
                return None
            # prefer frontiers inside the maze structure over open floor: the pillars stand in the maze
            walls = (self.g.logodds[r0:r1, c0:c1] > self.P["l_occ"]).astype(np.float32)
            w = max(int(self.P["frontier_structure_r"] / self.g.res), 1)
            dens = np.clip(ndimage.uniform_filter(walls, size=2 * w + 1) / 0.08, 0, 1)
            score = cum + (1 - dens) * self.P["frontier_structure_w"] / self.g.res
            # information: prefer frontiers next to space the lidar has never seen (a pillar can
            # only hide there - anywhere else the lidar would show it as a small round object)
            k = max(int(self.P["frontier_gain_r"] / self.g.res), 1)
            lid_known = ndimage.binary_closing(self.g.lidar_seen[r0:r1, c0:c1], iterations=3)
            unk = ndimage.uniform_filter((~lid_known).astype(np.float32), size=2 * k + 1)
            score = score - self.P["frontier_gain_w"] / self.g.res * unk
            if self.heading is not None and self.P["explore_forward_w"] > 0:
                # keep going the way the robot is driving (along a corridor) rather than turning
                # off sideways or back at every junction: half a turn costs explore_forward_w metres
                ang = np.arctan2(rr - s[0], cc - s[1])
                score = score + self.P["explore_forward_w"] / self.g.res * (1 - np.cos(ang - self.heading)) / 2
            if prefer is not None:
                # keep exploring where the last target was (no flapping between directions)
                score = score + self.P["explore_persist"] * np.hypot(rr - prefer[0], cc - prefer[1])
            bl = blob[r0:r1, c0:c1]
            score = score - self.P["blob_bonus"] / self.g.res * bl
            k = np.argmin(np.where(cand, score, np.inf))
            end = np.unravel_index(k, cum.shape)
            if bl[end]:
                self.last_tier = 0
            self.last_debug = (f"tier {tier} cands {int(cand.sum())} best score {score[end] * self.g.res:.2f} "
                               f"cum {cum[end] * self.g.res:.2f} min_dist {float(dist[cand].min()):.2f}")
            m.find_costs([(s[0] - r0, s[1] - c0)], ends=[end])
        else:
            ends = list(zip(gr - r0, gc - c0))
            ends = [e for e in ends if 0 <= e[0] < r1 - r0 and 0 <= e[1] < c1 - c0]
            cum, _ = m.find_costs([(s[0] - r0, s[1] - c0)], ends=ends)
            vals = np.array([cum[e] for e in ends])
            if not np.isfinite(vals).any():
                return None
            end = ends[int(np.argmin(vals))]
        path = m.traceback(end)
        return [(r + r0, c + c0) for r, c in path]

    def frontier_mask(self):
        """exploration target tiers, tried in this order:
        1. viewpoints of pillar candidates (small lidar objects the camera has not looked at)
        2. free cells next to space the lidar has never seen (a pillar can only hide there,
           because the lidar would show it as a small round object anywhere else)
        3. free cells next to floor the depth camera has not checked (floating walls, green)
        Tiers 2 and 3 are first restricted to the bounding box of the confirmed walls: the mazes
        are open, beyond the outermost walls there is only empty floor."""
        g, P = self.g, self.P
        free = g.seen & (g.logodds <= 0) & ~self.blocked      # floor seen by lidar or camera
        tiers = []
        if self.blob_targets is not None and self.blob_targets.any():
            tiers.append(self.blob_targets & free)
        # far lidar rays are sparser than the grid: close those small gaps before looking for unknown space
        lid_known = ndimage.binary_closing(g.lidar_seen, iterations=3)
        lid_front = free & ndimage.binary_dilation(~lid_known, iterations=1)
        cam_front = free & ndimage.binary_dilation(~g.cam_seen, iterations=1)
        walls = g.logodds > P["l_sticky"]
        box = None
        if walls.sum() > P["explore_box_min_cells"]:
            rows, cols = np.nonzero(walls)
            m = int(round(P["explore_box_margin"] / g.res))
            box = np.zeros_like(free)
            box[max(rows.min() - m, 0):rows.max() + m + 1, max(cols.min() - m, 0):cols.max() + m + 1] = True
        for fm in ((lid_front, cam_front) if P["explore_lidar_frontier"] else (None, cam_front)):
            if fm is None:
                tiers.append(np.zeros_like(free))
                continue
            tiers.append(fm & box if box is not None else fm)
        tiers.append(cam_front)
        return tiers


# Reference: the code in this file was generated with Claude (Anthropic LLM, claude.ai), Oct 2026,
# in a guided session for the Autonomous Robots Modularbeit (prompt: "implement navigation for maze 1").
