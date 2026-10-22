"""
Rosbot navigator for the Autonomous Robots Modularbeit: start -> blue pillar -> yellow pillar.

One code path for every maze:
  sense   lidar (walls), depth camera (floating / low walls), rgb camera (green floor, pillars),
          IMU yaw + wheel encoders (odometry)
  map     occupancy grid with separate layers for lidar walls, camera obstacles and green floor
  decide  goal = pillar if it has been seen, otherwise the cheapest exploration frontier
  plan    minimum-cost path on the inflated grid (unknown space allowed but more expensive)
  act     pure-pursuit path following, speed limited by clearance; back-up when stuck
"""
import math, os, re, sys, time, traceback, platform
import numpy as np
from scipy import ndimage
from controller import Robot

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import params as params_mod
from grid import Grid
from perception import Perception
from planner import Planner
from control import Follower, wheel_speeds, wrap
from localize import ScanMatcher, SlipCheck
from lidar_deskew import LidarDeskew

robot = Robot()
DT_MS = int(robot.getBasicTimeStep())
DT = DT_MS / 1000.0
P = params_mod.load(robot.getWorldPath())
DEBUG = os.environ.get("NAV_DEBUG", "")

# ------------------------------------------------------------------ devices
motors = [robot.getDevice(n) for n in ("fl_wheel_joint", "fr_wheel_joint", "rl_wheel_joint", "rr_wheel_joint")]
for m in motors:
    m.setPosition(float("inf")); m.setVelocity(0.0)
encs = [robot.getDevice(n) for n in ("front left wheel motor sensor", "front right wheel motor sensor",
                                     "rear left wheel motor sensor", "rear right wheel motor sensor")]
for e in encs:
    e.enable(DT_MS)
lidar = robot.getDevice("laser"); lidar.enable(DT_MS)
CAM_STEPS = 3
cam = robot.getDevice("camera rgb"); cam.enable(DT_MS * CAM_STEPS)
depth = robot.getDevice("camera depth"); depth.enable(DT_MS * CAM_STEPS)
imu = robot.getDevice("imu inertial_unit"); imu.enable(DT_MS)
irs = [robot.getDevice(n) for n in ("fl_range", "fr_range", "rl_range", "rr_range")]
for s_ in irs:
    s_.enable(DT_MS)

per = Perception(P, lidar.getHorizontalResolution(), cam.getWidth(), cam.getHeight(), cam.getFov(),
                 depth.getWidth(), depth.getHeight(), depth.getFov())
grid = Grid(P)
planner = Planner(grid, P)
follow = Follower(P)
matcher = ScanMatcher(grid, P)
slipper = SlipCheck(P)
near_hazard_n = [0]            # camera frames with a too-close slab edge (diagnostics)
deskew = LidarDeskew(lidar.getHorizontalResolution(), lidar.getFrequency())


IR_POSE = [(0.1, 0.05, 0.13), (0.1, -0.05, -0.13), (-0.1, 0.05, 3.01), (-0.1, -0.05, 3.27)]  # Rosbot.proto


def drive(v, w):
    """wheel commands with limited acceleration: stopping from full speed within one step tips
    the robot forward by up to 9 deg, and the tilted lidar then sees the floor as walls"""
    global cmd_v, cmd_w
    dv, dw = P["decel"] * DT, P["w_accel"] * DT
    v = min(max(v, cmd_v - dv), cmd_v + dv)
    w = min(max(w, cmd_w - dw), cmd_w + dw)
    cmd_v, cmd_w = v, w
    wl, wr = wheel_speeds(v, w, P)
    motors[0].setVelocity(wl); motors[2].setVelocity(wl)
    motors[1].setVelocity(wr); motors[3].setVelocity(wr)


def line_clear(occ, x0, y0, x1, y1, stop_short):
    """no occupied cell on the straight line from (x0, y0) towards (x1, y1), ending stop_short
    before (x1, y1)"""
    L = math.hypot(x1 - x0, y1 - y0) - stop_short
    if L <= 0:
        return True
    k = np.arange(0.0, L, grid.res * 0.7)
    ux, uy = (x1 - x0) / (L + stop_short), (y1 - y0) / (L + stop_short)
    r, c = grid.cell(x0 + ux * k, y0 + uy * k)
    ok = grid.inside(r, c)
    return not occ[r[ok], c[ok]].any()


def reach_mask(x, y, radius, exclude_bad=True, occ=None):
    """cells within `radius` of the pillar that also see it (no wall in between): 'reached'
    has to mean next to the pillar, not next to it on the other side of a wall"""
    m = disc_mask(x, y, radius)
    if exclude_bad:
        for bx, by in bad_approach.get(goals[gi] if gi < len(goals) else "", []):
            m &= ~disc_mask(bx, by, P["approach_exclude"])
    rr, cc = np.nonzero(m)
    if len(rr) == 0:
        return m
    if occ is None:
        occ = sight_blockers()
    xs, ys = (np.asarray(v, float) for v in grid.xy(rr, cc))
    d = np.hypot(x - xs, y - ys)
    stop = P["pillar_radius"] + P["pillar_los_clear"]
    f = np.linspace(0.0, 1.0, int(radius / (grid.res * 1.2)) + 2)[None, :]
    L = np.maximum(d - stop, 0.0)[:, None] / np.maximum(d, 1e-6)[:, None]
    px = xs[:, None] + (x - xs)[:, None] * f * L
    py = ys[:, None] + (y - ys)[:, None] * f * L
    r, c = grid.cell(px, py)
    r = np.clip(r, 0, grid.n - 1); c = np.clip(c, 0, grid.n - 1)
    blocked = occ[r, c].any(axis=1)
    m[rr[blocked], cc[blocked]] = False
    return m


def disc_mask(x, y, radius):
    gr, gc = grid.cell(x, y)
    rad = int(radius / grid.res) + 1
    rr, cc = np.mgrid[-rad:rad + 1, -rad:rad + 1]
    inside = np.hypot(rr, cc) * grid.res < radius
    mask = np.zeros_like(grid.seen)
    mask[np.clip(gr + rr[inside], 0, grid.n - 1), np.clip(gc + cc[inside], 0, grid.n - 1)] = True
    return mask


explore_target = None          # (row, col) of the exploration target we are committed to
last_explore = None            # the previous exploration target (new targets are preferred near it)


def update_blob_targets():
    """viewpoints (reachable free cells 0.6-1.5 m away) of the pillar candidates"""
    global blob_cands
    blob_cands = grid.pillar_candidates(known=[(v[1], v[2]) for v in pillars.values()] + looked_at)
    # single (not yet confirmed) sightings of the colour we are looking for are candidates too
    goal = goals[gi] if gi < len(goals) else None
    if goal is not None and goal not in pillars:
        for ox_, oy_, *_ in obs_hist.get(goal, []):
            if all(math.hypot(ox_ - lx, oy_ - ly) > 0.4 for lx, ly in looked_at) and \
                    all(math.hypot(ox_ - bx, oy_ - by) > 0.2 for bx, by, _, _ in blob_cands):
                r, c = grid.cell(ox_, oy_)
                sl = (slice(r, r + 1), slice(c, c + 1))
                blob_cands.append((float(ox_), float(oy_), sl, np.ones((1, 1), bool)))
    if not blob_cands:
        planner.blob_targets = None
        return
    # viewpoints: 0.6-1.5 m away AND with a clear line of sight to the candidate
    occ = sight_blockers()
    tg = np.zeros((grid.n, grid.n), bool)
    for x, y, sl, m in blob_cands:
        tg |= reach_mask(x, y, P["blob_view_max"], exclude_bad=False, occ=occ) & ~disc_mask(x, y, P["blob_view_min"])
    planner.blob_targets = tg


def blob_near(r, c):
    x, y = grid.xy(r, c)
    return any(math.hypot(bx - x, by - y) <= P["blob_view_max"] + 0.1 for bx, by, _, _ in blob_cands)


def arrived_at_target():
    """at an exploration target: remember the area as visited and look around once"""
    global spin_left
    if explore_tier == 0:
        # viewpoint of a pillar candidate: look around (the candidate is then 'looked at')
        for bx, by, sl, m in blob_cands:
            if math.hypot(bx - pose[0], by - pose[1]) <= P["blob_view_max"] + 0.2:
                grid.looked[sl][m] = True
                looked_at.append((bx, by))
        r0, c0 = grid.cell(pose[0], pose[1])
        if planner.d_obst is not None and planner.d_obst[r0, c0] >= P["spin_clearance"]:
            spin_left = 2 * math.pi
            log(f"[nav] t={t:6.2f}s at a pillar candidate's viewpoint -> looking around")
        return
    k = int(P["visited_radius"] / grid.res)
    r0, c0 = grid.cell(pose[0], pose[1])
    yy, xx = np.mgrid[-k:k + 1, -k:k + 1]
    m = np.hypot(yy, xx) <= k
    grid.cam_seen[np.clip(r0 + yy[m], 0, grid.n - 1), np.clip(c0 + xx[m], 0, grid.n - 1)] = True
    # spinning costs >1 s: only worth it if much of the space visible from here is still unchecked
    k2 = int(P["spin_look_radius"] / grid.res)
    sl = (slice(max(r0 - k2, 0), r0 + k2 + 1), slice(max(c0 - k2, 0), c0 + k2 + 1))
    visible = grid.seen[sl] & (grid.logodds[sl] < 0)
    unchecked = (visible & ~grid.cam_seen[sl]).sum() / max(int(visible.sum()), 1)
    if planner.d_obst is not None and planner.d_obst[r0, c0] >= P["spin_clearance"] and unchecked > P["spin_min_unchecked"]:
        spin_left = 2 * math.pi
        log(f"[nav] t={t:6.2f}s exploration target reached -> looking around ({unchecked:.0%} unchecked)")


def explore(rc):
    """path to the committed exploration target; a new target is picked only when the old one
    is reached, unreachable, or has been checked by the camera in the meantime"""
    global explore_target, explore_tier, last_explore
    update_blob_targets()
    # exploration also considers gaps that are only passable with the smallest margin, at a price,
    # so that a narrow entrance into unexplored maze is not ignored for far-away open floor
    planner.build(inflate=P["fallback_inflation"][-1])
    try:
        return _explore(rc)
    finally:
        planner.build()


def _explore(rc):
    global explore_target, explore_tier, last_explore
    if explore_target is not None:
        r, c = explore_target
        here = math.hypot(r - rc[0], c - rc[1]) * grid.res
        blk = (slice(max(r - 3, 0), r + 4), slice(max(c - 3, 0), c + 4))
        if explore_tier == 0:
            checked = not blob_near(r, c)
        elif explore_tier == 1:
            checked = grid.lidar_seen[blk].all()
        else:
            checked = grid.cam_seen[blk].all()
        if here < P["explore_reached"]:
            arrived_at_target()
            explore_target = None
            if spin_left > 0:
                return None          # looking around first
            # otherwise continue straight to the next target (no stop in between)
        elif checked or planner.blocked[r, c]:
            explore_target = None
        else:
            x, y = grid.xy(r, c)
            path = planner.plan(rc, goal_mask=disc_mask(float(x), float(y), 2.5 * grid.res))
            if path is not None:
                return path
            explore_target = None
    planner.heading = pose[2]
    path = planner.plan(rc, frontier=True, prefer=last_explore)
    if path is None:
        # narrow gaps: the same smaller margins as for a known goal
        for infl, conf in [(i, False) for i in P["fallback_inflation"]] + ([(P["fallback_inflation"][-1], True)] if P["fallback_confident"] else []):
            planner.build(inflate=infl, confident=conf)
            path = planner.plan(rc, frontier=True, prefer=last_explore)
            if path is not None:
                log(f"[nav] t={t:6.2f}s exploration route only with margin {infl:.3f} m"
                    + (" using confirmed walls only" if conf else ""), echo=False)
                break
    if path is not None:
        explore_target = tuple(int(v) for v in path[-1])
        explore_tier = planner.last_tier
        if explore_tier != 0:
            last_explore = explore_target    # a candidate detour does not move the exploration focus
        x, y = grid.xy(*explore_target)
        kind = ["pillar candidate", "lidar frontier", "camera frontier", "camera frontier (outside)"][explore_tier or 0]
        log(f"[nav] t={t:6.2f}s exploring towards ({float(x):.2f},{float(y):.2f}) [{kind}] {getattr(planner, 'last_debug', '')}", echo=False)
        if DEBUG and math.hypot(float(x) - pose[0], float(y) - pose[1]) > 2.5:
            save_debug(f"far_{t:.2f}")
    return path


def sight_weight(dist):
    """reliability of a sighting: full up to sight_full_dist, then falling with distance squared
    (range and bearing errors, and small pose errors, grow with the distance)"""
    return min(1.0, (P["sight_full_dist"] / max(dist, 1e-3)) ** 2)


def fuse_pillar(col):
    """densest cluster of agreeing sightings -> [n, x, y], or None if too few agree"""
    pts = np.array(obs_hist[col])
    if len(pts) < 2:
        return None
    d = np.hypot(pts[:, None, 0] - pts[None, :, 0], pts[:, None, 1] - pts[None, :, 1])
    support = (d < P["pillar_cluster"]).sum(axis=1)
    # the cluster used is the one with the most WEIGHTED support: close-up sightings count more
    wsup = ((d < P["pillar_cluster"]) * pts[None, :, 2]).sum(axis=1)
    k = int(np.argmax(wsup))
    if support[k] < P["pillar_min_obs"]:
        # two sightings that agree very closely are enough too (a pillar glimpsed briefly while
        # turning, e.g. right next to the start)
        tight = (d < P["pillar_tight"]).sum(axis=1)
        k = int(np.argmax(tight))
        if tight[k] < 2:
            return None
        members = pts[d[k] < P["pillar_tight"]]
        return [int(tight[k]), float(np.median(members[:, 0])), float(np.median(members[:, 1]))]
    members = pts[d[k] < P["pillar_cluster"]]
    return [int(support[k]), float(np.median(members[:, 0])), float(np.median(members[:, 1]))]


def pillar_confirmed(px, py, col):
    """'reached' only counts if the camera saw the pillar, with a distance from a range sensor
    (lidar or depth camera), at the estimated place within the last moment. A wall right where
    the estimate is (pillar glimpsed through a gap) or an empty spot cannot pass this."""
    if col not in last_ranged:
        return False
    t0, ox, oy = last_ranged[col]
    return t - t0 < P["confirm_recent"] and math.hypot(ox - px, oy - py) < P["pillar_cluster"]


def check_reached(col, px, py):
    """close to the estimate: 'reached' (seen there just now), 'facing' (turning so that the
    camera can check), 'absent' (in plain view but not there) or 'hidden' (a wall is in between,
    so nothing can be concluded from here)"""
    global face_until, face_xy, backoff_until, backoff_count
    if pillar_confirmed(px, py, col):
        return "reached"
    if not line_clear(sight_blockers(), pose[0], pose[1], px, py, P["pillar_radius"] + P["pillar_los_clear"]):
        return "hidden"
    bearing = wrap(math.atan2(py - pose[1], px - pose[0]) - pose[2])
    if abs(bearing) > cam.getFov() / 2 - P["look_fov_margin"] and t > face_until + 1.0:
        face_until, face_xy = t + P["face_time"], (px, py)
    if t < face_until:
        drive(0.0, float(np.clip(3.0 * bearing, -P["w_max"], P["w_max"])))
        return "facing"
    # very close up neither camera nor lidar can see the pillar reliably: back off and look again
    d = math.hypot(px - pose[0], py - pose[1])
    if d < P["confirm_min_dist"] and t > backoff_until + P["backoff_wait"] and backoff_count < 2:
        backoff_until = t + P["backoff_time"]
        backoff_count += 1
    if t < backoff_until:
        drive(-P["backup_speed"], float(np.clip(2.0 * bearing, -1.5, 1.5)))
        return "facing"
    # the lidar must see free space up to the pillar: a nearer return along that bearing means a
    # wall (with a slit the map did not show) is in between - then nothing can be concluded
    xr, yr = px - pose[0], py - pose[1]
    c_, s_ = math.cos(-pose[2]), math.sin(-pose[2])
    bx_, by_ = c_ * xr - s_ * yr - 0.02, s_ * xr + c_ * yr
    dl = math.hypot(bx_, by_)
    r_l, _, _ = per.lidar_range_at(lid_now, math.atan2(by_, bx_) + 0.05, math.atan2(by_, bx_) - 0.05)
    if r_l is not None and r_l < dl - P["pillar_radius"] - 0.06:
        return "hidden"
    return "absent"


def sight_blockers():
    """obstacles for line-of-sight tests: the map with slits between wall ends closed, and
    space no sensor has observed (it may hide the rest of a wall)"""
    occ = grid.occupied() | ~grid.seen
    k = int(round(P["wall_close_gap"] / 2 / grid.res))
    yy, xx = np.mgrid[-k:k + 1, -k:k + 1]
    return ndimage.binary_closing(occ, structure=(yy ** 2 + xx ** 2) <= k * k)


def near_bad_approach(col):
    return any(math.hypot(bx - pose[0], by - pose[1]) < P["approach_exclude"] for bx, by in bad_approach.get(col, []))


def failed_approach(col, t):
    """pillar not seen although we are at its estimated place. With a well-supported estimate
    this is usually an approach from the wrong side of a wall (slit, pose error): exclude this
    spot as an approach point and try another side; discard the estimate only after repeated
    failures or if it rests on few sightings"""
    global last_plan_t
    est = pillars[col]
    bad_approach.setdefault(col, []).append((pose[0], pose[1]))
    if est[0] < P["approach_keep_support"] or len(bad_approach[col]) > P["approach_tries"]:
        bad_approach[col] = []
        reject_estimate(col, t)
        return
    follow.set_path(None)
    last_plan_t = -1.0
    log(f"[nav] t={t:6.2f}s {col} pillar not visible from here - trying another approach ({len(bad_approach[col])})")


def reject_estimate(col, t):
    """we are where the pillar should be, but it is not there: forget those sightings"""
    global last_plan_t
    global backoff_count
    backoff_count = 0
    est = pillars.pop(col)
    obs_hist[col] = [p for p in obs_hist[col] if math.hypot(p[0] - est[1], p[1] - est[2]) > 0.4]
    follow.set_path(None)
    last_plan_t = -1.0
    log(f"[nav] t={t:6.2f}s {col} pillar NOT found at ({est[1]:.2f},{est[2]:.2f}) - estimate discarded, searching again")


def goal_reached(goal, t, note=""):
    """bookkeeping when a pillar is reached; returns the index of the next goal"""
    global gi, last_plan_t, last_explore, explore_target, backoff_count
    backoff_count = 0
    t_reached[goal] = t
    last_explore = explore_target = None
    log(f"[nav] t={t:6.2f}s REACHED {goal}{note}")
    save_debug(f"reached_{goal}")
    gi += 1
    follow.set_path(None)
    last_plan_t = -1.0
    if gi >= len(goals):
        log(f"[nav] FINISHED: start->blue {t_reached['blue']:.2f}s, "
              f"blue->yellow {t_reached['yellow'] - t_reached['blue']:.2f}s, total {t_reached['yellow']:.2f}s")
        log("[nav] speed limited by: " + ", ".join(f"{k} {v:.1f}s" for k, v in
              sorted(follow.why.items(), key=lambda kv: -kv[1])))


def to_world(pose, xr, yr):
    x, y, th = pose
    c, s = math.cos(th), math.sin(th)
    return x + c * xr - s * yr, y + s * xr + c * yr


# ------------------------------------------------------------------ state
pose = [0.0, 0.0, 0.0]
enc_prev = None
pillars = {}                  # colour -> [n_agreeing, x, y] once enough sightings agree
obs_hist = {}                 # colour -> list of observed (x, y)
last_ranged = {}
face_until = -10.0            # turning towards a pillar estimate until then
backoff_until = -10.0         # backing away from a too-close pillar estimate until then
backoff_count = 0             # back-offs for the current estimate
bad_approach = {}             # colour -> robot positions from which the pillar could not be seen
face_xy = (0.0, 0.0)
explore_tier = None           # which kind of exploration target is being driven to
looked_at = []                # pillar candidates already looked at from close by
blob_cands = []               # pillar candidates (small lidar objects not yet looked at)              # colour -> (t, x, y) of the last sighting with a range-sensor distance
goals = ["blue", "yellow"]
gi = 0
t_reached = {}
step = 0
last_plan_t = -1.0
progress_ref = (0.0, 0.0, 0.0, 0.0)  # (t, x, y, yaw)
cmd_activity = 0
backup_until = -1.0
last_match_xy = [0.0, 0.0]    # pose at the previous scan match (for the odometry step)
cmd_v, cmd_w = 0.0, 0.0       # last commanded speeds (for the acceleration limit)
stuck_count = 0
stuck_where = (1e9, 1e9)
# escape manoeuvres (forward speed sign, turn sign), tried in turn when stuck again at the same spot
RECOVERY = [(-1, 0), (-1, 1), (-1, -1), (-1, 0), (0, 1), (0, -1)]
plan_fail = 0
next_dbg = 0.0
looked_around = set()          # goals for which the look-around spin was done
spin_stall = [0.0]             # time the look-around spin has not been turning [s]
spin_left = 0.0                # remaining angle of the current look-around spin [rad]

# ------------------------------------------------------------------ run log (always on)
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
WORLD = os.path.splitext(os.path.basename(robot.getWorldPath() or "world"))[0]
_logf = open(os.path.join(LOG_DIR, f"{WORLD}_{time.strftime('%Y%m%d-%H%M%S')}.log"), "w")


def log(msg, echo=True):
    """every message goes to the log file; the Webots console shows only the main events"""
    _logf.write(msg + "\n"); _logf.flush()
    if echo:
        line = console_line(msg)
        if line:
            print(line, flush=True)


def console_line(msg):
    """short console form of a log message, or None for details that stay in the log file"""
    if "CONTROLLER ERROR" in msg:
        return msg
    m = re.match(r"\[nav\] world (\S+),", msg)
    if m:
        return f"Rosbot navigation | {m.group(1)} | target: blue pillar, then yellow pillar"
    m = re.match(r"\[nav\] t=\s*([\d.]+)s located (\w+) pillar", msg)
    if m:
        return f"t = {float(m.group(1)):6.2f} s | {m.group(2).capitalize()} pillar located"
    m = re.match(r"\[nav\] t=\s*([\d.]+)s REACHED (\w+)", msg)
    if m:
        return f"t = {float(m.group(1)):6.2f} s | {m.group(2).capitalize()} pillar REACHED"
    m = re.match(r"\[nav\] FINISHED: start->blue ([\d.]+)s, blue->yellow ([\d.]+)s, total ([\d.]+)s", msg)
    if m:
        return ("RESULT\n"
                f"  start -> blue pillar  : {float(m.group(1)):6.2f} s\n"
                f"  blue  -> yellow pillar: {float(m.group(2)):6.2f} s\n"
                f"  total                 : {float(m.group(3)):6.2f} s")
    return None


def versions():
    out = [f"python {platform.python_version()} on {platform.platform()}", f"numpy {np.__version__}"]
    for mod in ("cv2", "scipy", "skimage"):
        try:
            out.append(f"{mod} {__import__(mod).__version__}")
        except Exception as ex:  # pragma: no cover
            out.append(f"{mod} MISSING ({ex})")
    return ", ".join(out)


log(f"[nav] world {WORLD}, parameters: {P['_param_file']}")
log(f"[nav] {versions()}")
log(f"[nav] timestep {DT_MS} ms, lidar {lidar.getHorizontalResolution()} beams {lidar.getFrequency()} Hz, "
    f"camera {cam.getWidth()}x{cam.getHeight()}, depth {depth.getWidth()}x{depth.getHeight()}")
mode = "start"
next_status = 0.0
snap_done = False


def save_debug(tag):
    if not DEBUG:
        return
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    img = np.zeros((grid.n, grid.n, 3), np.float32) + 0.55
    img[grid.seen] = 1.0
    img[grid.seen & ~grid.band_ok] = (0.85, 0.85, 1.0)
    img[grid.logodds > P["l_occ"]] = (0.35, 0.25, 0.15)
    img[grid.vis_obst] = (0.85, 0.1, 0.1)
    img[grid.green] = (0.1, 0.75, 0.1)
    fig, ax = plt.subplots(figsize=(7, 7))
    ext = [-grid.off * grid.res, (grid.n - grid.off) * grid.res] * 2
    ax.imshow(img, origin="lower", extent=ext)
    if follow.path is not None:
        ax.plot(follow.path[:, 0], follow.path[:, 1], "m-", lw=1.5)
    ax.plot(pose[0], pose[1], "ko")
    for k, v in pillars.items():
        ax.plot(v[1], v[2], "o", color="b" if k == "blue" else "y", ms=9, mec="k")
    rows, cols = np.nonzero(grid.seen)
    if len(rows):
        xa, ya = grid.xy(rows.min(), cols.min()); xb, yb = grid.xy(rows.max(), cols.max())
        ax.set_xlim(xa - 0.5, xb + 0.5); ax.set_ylim(ya - 0.5, yb + 0.5)
    ax.set_title(f"{tag} t={robot.getTime():.1f}s")
    fig.savefig(os.path.join(DEBUG, f"map_{tag}.png"), dpi=90); plt.close(fig)
    np.savez_compressed(os.path.join(DEBUG, f"grid_{tag}.npz"), logodds=grid.logodds, vis=grid.vis_obst, green=grid.green, seen=grid.seen, cam_seen=grid.cam_seen, band_ok=grid.band_ok, swept=grid.swept, trail=grid.trail, lidar_seen=grid.lidar_seen, looked=grid.looked, shadow_ray=grid.shadow_ray, green_ev=grid.green_ev, pose=np.array(pose))


def step_once():
    global step, t, pose, enc_prev, gi, last_plan_t, progress_ref, cmd_activity, backup_until, plan_fail
    global stuck_count, stuck_where
    global next_dbg, spin_left, mode, next_status, snap_done, lid_full, lid_now
    step += 1
    t = robot.getTime()
    # ---------------------------------------------------------- odometry
    roll, pitch, yaw = imu.getRollPitchYaw()
    e = [s.getValue() for s in encs]
    if enc_prev is None:
        enc_prev = e
    ds = P["wheel_radius"] * float(np.mean([a - b for a, b in zip(e, enc_prev)]))
    enc_prev = e
    if max(abs(pitch), abs(roll)) > P["odo_max_tilt"]:
        # strongly tilted (reared up under a slab, hanging on an edge): the wheels spin or lose
        # ground contact - the encoders no longer measure motion over the floor
        ds = 0.0
    yaw_rate = wrap(yaw - pose[2]) / DT
    th_mid = math.atan2(math.sin(pose[2]) + math.sin(yaw), math.cos(pose[2]) + math.cos(yaw))
    pose[0] += ds * math.cos(th_mid); pose[1] += ds * math.sin(th_mid); pose[2] = yaw
    grid.mark_robot(pose)
    # ---------------------------------------------------------- lidar mapping
    deskew.add_pose(t, pose[0], pose[1], pose[2])
    raw = lidar.getRangeImage()
    bx, by, byaw = deskew.beam_poses(t, raw)          # robot pose at each beam's measurement time
    lid_full = np.asarray(raw, float)
    lid_full[lid_full < P["lidar_min_range"]] = np.nan      # 0 = not measured yet (first sweep)
    ok = np.isfinite(lid_full)
    # when the robot pitches/rolls (accelerating, braking, bumps) the lidar plane is tilted and
    # far beams hit the floor; those returns would be mapped as phantom walls. A beam tilted by
    # e rad reaches the floor at h / e, so returns beyond a safe share of that are dropped.
    tilt = np.abs(pitch * np.cos(per.lidar_ang)) + np.abs(roll * np.sin(per.lidar_ang))
    floor_r = P["lidar_height"] / np.maximum(tilt, 1e-4)
    ok &= ~(lid_full > P["floor_hit_share"] * floor_r)
    rng = lid_full[ok]
    # the same scan re-indexed by the bearing relative to the CURRENT heading: the rotating
    # lidar needs 83 ms per sweep, while turning on the spot the robot turns up to 0.3 rad
    # meanwhile, so raw beam index and bearing no longer match (pillar ranging, line of sight)
    nb = len(lid_full)
    a_now = np.mod(byaw + per.lidar_ang - pose[2] + math.pi, 2 * math.pi) - math.pi
    j = np.mod(np.round((math.pi - a_now) / (2 * math.pi) * nb - 0.5).astype(int), nb)
    lid_now = np.full(nb, np.nan)
    fin = np.isfinite(lid_full)
    order = np.argsort(-np.where(fin, lid_full, -1.0))          # nearer readings written last (win)
    lid_now[j[order]] = lid_full[order]
    lid_now[~np.isfinite(lid_now)] = np.nan
    a_w = byaw[ok] + per.lidar_ang[ok]
    ox, oy = bx[ok] + 0.02 * np.cos(byaw[ok]), by[ok] + 0.02 * np.sin(byaw[ok])
    if step % int(P["match_every"]) == 0:
        matcher.refresh(t)
        odo_step = (pose[0] - last_match_xy[0], pose[1] - last_match_xy[1])
        (dx, dy), gain = matcher.match(ox, oy, a_w, rng, odo_step)
        if gain > 0.0:
            pose[0] += dx; pose[1] += dy; ox = ox + dx; oy = oy + dy
            deskew.hist[-1] = (t, pose[0], pose[1], deskew.hist[-1][3])
        if step % int(P["slip_every"]) == 0 and abs(yaw_rate) < P["map_max_turn_rate"] and abs(pitch) < P["slip_max_tilt"] and abs(roll) < P["slip_max_tilt"]:
            sdx, sdy = slipper.check(t, pose, ox, oy, a_w, rng)
            if sdx or sdy:
                # the map built so far must agree: in rows of parallel boards an earlier scan
                # also fits after a shift by one board spacing, which is not slip
                fits = matcher.fit(ox, oy, a_w, rng, [(0.0, 0.0), (sdx, sdy)])
                if fits is not None and fits[1] > P["slip_map_ratio"] * fits[0]:
                    log(f"[nav] t={t:6.2f}s wheel-slip correction of {math.hypot(sdx, sdy):.2f} m rejected: the map does not agree")
                    sdx = sdy = 0.0
            if sdx or sdy:
                pose[0] += sdx; pose[1] += sdy; ox = ox + sdx; oy = oy + sdy
                deskew.hist[-1] = (t, pose[0], pose[1], deskew.hist[-1][3])
                log(f"[nav] t={t:6.2f}s wheel slip: scans show {math.hypot(sdx, sdy):.2f} m less motion than odometry -> pose corrected")
        last_match_xy[:] = [pose[0], pose[1]]
        # while turning fast, small timing errors smear walls sideways (making them longer and
        # closing gaps): only map when the robot turns slowly; scan matching still runs
        # ... and not while tilted (wedged / climbing): the lidar plane then cuts the floor
        if abs(yaw_rate) < P["map_max_turn_rate"] and abs(pitch) < P["map_max_tilt"] and abs(roll) < P["map_max_tilt"]:
            grid.update_lidar(ox, oy, a_w, rng)
        inf = np.isinf(lid_full)
        if inf.any():
            grid.mark_open_rays(bx[inf], by[inf], byaw[inf] + per.lidar_ang[inf])
    # ---------------------------------------------------------- infra-red (front blind zone, low obstacles)
    for (sx, sy, sa), s_ in zip(IR_POSE[:2], irs[:2]):
        val = s_.getValue()
        if val < P["ir_use_max"]:
            wx, wy = to_world(pose, np.array([sx + val * math.cos(sa)]), np.array([sy + val * math.sin(sa)]))
            grid.add_obstacle(wx, wy, weight=P["vis_hits"])
    # ---------------------------------------------------------- cameras
    if step % CAM_STEPS == 0:
        hsv = per.hsv(cam.getImage())
        dimg = per.depth_array(depth.getRangeImage(data_type="buffer"))
        (gx, gy), (nx, ny) = per.green_points(hsv, dimg)
        steady = abs(pitch) < P["cam_max_tilt"] and abs(roll) < P["cam_max_tilt"] and abs(yaw_rate) < P["cam_max_turn_rate"]
        if (len(gx) or len(nx)) and abs(yaw_rate) < P["cam_max_turn_rate"]:
            wx, wy = to_world(pose, gx, gy); wnx, wny = to_world(pose, nx, ny)
            fresh = grid.update_green(wx, wy, wnx, wny)
            # green newly mapped close to the path ahead -> replan right away
            if follow.path is not None and fresh.any():
                seg = follow.path[follow.idx:follow.idx + 80]
                fr, fc = np.nonzero(fresh)
                if len(seg) and len(fr):
                    fxw, fyw = grid.xy(fr, fc)
                    dmin = np.min(np.hypot(seg[:, None, 0] - np.asarray(fxw)[None, ::2], seg[:, None, 1] - np.asarray(fyw)[None, ::2]))
                    if dmin < P["inflate_green"] + 0.05:
                        last_plan_t = -1.0
        if steady:
            (cx, cy), (fx, fy) = per.obstacle_points(dimg)
            if len(cx):
                wx, wy = to_world(pose, cx, cy); grid.add_obstacle(wx, wy, skip_near_lidar=True)
                # what lies right behind a seen floating edge (along the camera ray) is unknown at
                # camera-housing height: a slab may continue there
                cx, cy = per.float_pts
                dx_, dy_ = cx - 0.0, cy - 0.037
                nrm = np.maximum(np.hypot(dx_, dy_), 1e-6)
                ks = np.arange(1, int(P["shadow_dist"] / 0.03) + 1) * 0.03
                sx_ = (cx[:, None] + dx_[:, None] / nrm[:, None] * ks[None, :]).ravel()
                sy_ = (cy[:, None] + dy_[:, None] / nrm[:, None] * ks[None, :]).ravel()
                swx, swy = to_world(pose, sx_, sy_)
                grid.mark_shadow(swx, swy)
                # new obstacle close to the path ahead? -> replan right away
                if follow.path is not None:
                    seg = follow.path[follow.idx:follow.idx + 80]
                    vis = grid.vis_obst
                    rr, cc = grid.cell(wx, wy); okk = grid.inside(rr, cc)
                    pts = np.c_[wx[okk], wy[okk]][vis[rr[okk], cc[okk]]]
                    if len(pts) and len(seg):
                        dmin = np.min(np.hypot(seg[:, None, 0] - pts[None, ::4, 0], seg[:, None, 1] - pts[None, ::4, 1]))
                        if dmin < P["inflate_obst"] + 0.03:
                            last_plan_t = -1.0
            if len(fx):
                wx, wy = to_world(pose, fx, fy); grid.mark_free_points(wx, wy)
            # board edge closer than the depth camera can range, above the lidar plane: the lidar
            # must see nothing that close in that direction (else it is a normal, mapped wall).
            # Its distance is unknown (< 0.6 m): the cells along the ray up to that range are marked
            hz_du = per.near_hazard(hsv, dimg)
            if len(hz_du):
                keep = []
                for du_ in hz_du:
                    b_ = math.atan(-float(du_))
                    # nearest lidar return in a window around the bearing (the colour and depth
                    # cameras sit a few cm from the lidar: close wall ends shift by several degrees)
                    lo = int(math.floor((math.pi - b_ - P["hz_excl"]) / (2 * math.pi) * len(lid_now)))
                    hi = int(math.ceil((math.pi - b_ + P["hz_excl"]) / (2 * math.pi) * len(lid_now)))
                    win = lid_now[np.arange(lo, hi + 1) % len(lid_now)]
                    win = win[np.isfinite(win)]
                    if len(win) >= 3 and win.min() > P["hz_far"] / math.cos(b_) + P["hz_lidar_margin"]:
                        keep.append(float(du_))
                # a slab edge is a contiguous run of image columns; isolated columns are texture
                # seams or wall ends seen with a little parallax
                if keep:
                    step_ = float(np.median(np.diff(per.du[0]))) if per.du.shape[1] > 1 else 1.0
                    kk = np.array(sorted(keep))
                    run_id = np.r_[0, np.cumsum(np.diff(kk) > 1.5 * step_)]
                    sizes = np.bincount(run_id)
                    keep = [float(v) for v, r_ in zip(kk, run_id) if sizes[r_] >= P["hz_min_cols"]]
                if keep:
                    ks = np.arange(P["hz_near"], P["hz_far"] + 1e-9, grid.res)
                    du_ = np.array(keep)[:, None]
                    hx = np.broadcast_to(ks[None, :], (len(keep), len(ks))).ravel()
                    hy = (-du_ * ks[None, :] + 0.037).ravel()
                    wx, wy = to_world(pose, hx, hy)
                    grid.add_hazard(wx, wy)
                    near_hazard_n[0] += 1
                    if DEBUG:
                        print(f"[hazard] t={t:.2f} cols={len(keep)} bearings={math.degrees(math.atan(-keep[0])):.0f}..{math.degrees(math.atan(-keep[-1])):.0f}")
            bx_, by_ = per.band_clear_points(dimg)
            if len(bx_):
                wx, wy = to_world(pose, bx_, by_); grid.mark_band_clear(wx, wy)
            if grid.hz.any():
                bx_, by_ = per.band_clear_points(dimg, P["hz_clear_lo"], P["hz_clear_hi"])
                if len(bx_):
                    wx, wy = to_world(pose, bx_, by_); grid.clear_hazard(wx, wy)
        colour_spans = []
        for col in ("blue", "yellow"):
            obs = per.pillar(hsv, col, dimg, lid_now)
            if obs is None and per.seen_span is not None:
                colour_spans.append(per.seen_span)
            if obs is None:
                if DEBUG and per.reject:
                    log(f"[obs] t={t:.2f} {col} rejected: {per.reject}", echo=False)
                continue
            xr, yr, dist = obs
            wx, wy = to_world(pose, xr, yr)
            if DEBUG:
                log(f"[obs] t={t:.2f} {col} {per.last_method} dist={dist:.2f} at ({wx:.3f},{wy:.3f}) pose=({pose[0]:.3f},{pose[1]:.3f},{pose[2]:.3f})", echo=False)
            if per.last_method in ("depth+lidar", "lidar+width", "lidar-arc", "near-width"):
                last_ranged[col] = (t, float(wx), float(wy))
            if per.last_method == "lidar-arc":
                continue        # close-up border case: only confirms an estimate, never creates one
            obs_hist.setdefault(col, []).append((wx, wy, sight_weight(dist)))
            if len(obs_hist[col]) > P["pillar_obs_keep"]:
                # drop the least reliable sighting (the farthest; among equals the oldest), so a
                # few close-up sightings are not pushed out by many far, less precise ones
                h_ = obs_hist[col]
                del h_[min(range(len(h_)), key=lambda i: (h_[i][2], i))]
            est = fuse_pillar(col)
            if est is not None:
                first = col not in pillars
                old_ = pillars.get(col)
                pillars[col] = est
                if first:
                    log(f"[nav] t={t:6.2f}s located {col} pillar at ({est[1]:.2f},{est[2]:.2f}) from {est[0]} agreeing sightings")
                elif math.hypot(est[1] - old_[1], est[2] - old_[2]) > 0.2:
                    log(f"[nav] t={t:6.2f}s {col} pillar estimate moved to ({est[1]:.2f},{est[2]:.2f}) (sighting {per.last_method}, {dist:.2f} m away)")
        # lidar hits inside the camera view and range count as 'looked at' (a pillar there would
        # have been recognised) - except where pillar colour was seen but the sighting rejected
        half_fov = cam.getFov() / 2 - P["look_fov_margin"]
        la = a_now[ok]
        inview = (np.abs(la) < P["look_half_angle"]) & (rng < P["look_range"]) & (rng > P["look_min_range"])
        for bl, br in colour_spans:
            inview &= ~((la <= bl + 0.05) & (la >= br - 0.05))
        if inview.any():
            grid.mark_looked(ox[inview] + np.cos(a_w[inview]) * rng[inview], oy[inview] + np.sin(a_w[inview]) * rng[inview])
    # ---------------------------------------------------------- goal handling
    if gi >= len(goals):           # finished: stay where we are
        mode = "finished"
        drive(0, 0)
        return
    goal = goals[gi]
    if goal in pillars:
        gx_, gy_ = pillars[goal][1], pillars[goal][2]
        if math.hypot(gx_ - pose[0], gy_ - pose[1]) < P["reach_dist"] and not near_bad_approach(goal):
            verdict = check_reached(goal, gx_, gy_)
            if verdict == "reached":
                goal_reached(goal, t)
                return
            if verdict == "absent":
                failed_approach(goal, t)
                return
            if verdict == "facing":
                return
    # ---------------------------------------------------------- look around if the goal is unknown
    if goal not in pillars and goal not in looked_around and step > 2 * CAM_STEPS:
        looked_around.add(goal)
        r_, c_ = grid.cell(pose[0], pose[1])
        room = planner.d_obst is None or planner.d_obst[r_, c_] >= P["spin_clearance"]
        spin_left = 2 * math.pi if room else 0.0
        log(f"[nav] t={t:6.2f}s {goal} pillar unknown -> " + ("looking around" if spin_left > 0 else "no room to turn on the spot, exploring"))
    if spin_left > 0 and goal not in pillars:
        mode = "look-around"
        spin_left -= abs(yaw_rate) * DT
        # cannot turn (wedged under a slab, pressed against a wall): give up the spin and back out
        spin_stall[0] = spin_stall[0] + DT if abs(yaw_rate) < P["spin_stall_rate"] else 0.0
        if spin_stall[0] > P["stuck_time"]:
            spin_left, spin_stall[0] = 0.0, 0.0
            stuck_count += 1
            stuck_where = (pose[0], pose[1])
            backup_until = t + P["backup_time"] * (1.0 + 0.5 * min(stuck_count - 1, 3))
            log(f"[nav] t={t:6.2f}s cannot turn on the spot -> escape manoeuvre")
            return
        drive(0.0, P["look_around_rate"])
        return
    spin_left = 0.0
    # ---------------------------------------------------------- planning
    if t - last_plan_t >= P["replan_period"]:
        last_plan_t = t
        planner.build(clear_disc=(pillars[goal][1], pillars[goal][2], P["pillar_radius"] + P["pillar_clear"]) if goal in pillars else None)
        rc = tuple(int(v) for v in grid.cell(pose[0], pose[1]))
        path = None
        if goal in pillars:
            gx_, gy_ = pillars[goal][1], pillars[goal][2]
            for ring in (P["reach_dist"] - 0.03, P["reach_dist"] + P["reach_slack"]):
                path = planner.plan(rc, goal_mask=reach_mask(gx_, gy_, ring))
                if path is not None:
                    break
            d_goal = math.hypot(gx_ - pose[0], gy_ - pose[1])
            if (path is None or ring > P["reach_dist"]) and d_goal < P["reach_dist"] + P["reach_slack"] \
                    and not near_bad_approach(goal):
                verdict = check_reached(goal, gx_, gy_)
                if verdict == "reached":
                    goal_reached(goal, t, f" (as close as the surroundings allow: {d_goal:.2f} m)")
                    return
                if verdict == "absent":
                    failed_approach(goal, t)
                    return
                if verdict == "facing":
                    return
        if path is None and goal in pillars:
            # no route with the normal margin: retry with slightly smaller margins before giving up
            for infl, conf in [(i, False) for i in P["fallback_inflation"]] + ([(P["fallback_inflation"][-1], True)] if P["fallback_confident"] else []):
                planner.build(clear_disc=(gx_, gy_, P["pillar_radius"] + P["pillar_clear"]), inflate=infl, confident=conf)
                path = planner.plan(rc, goal_mask=reach_mask(gx_, gy_, P["reach_dist"] + P["reach_slack"]))
                if path is not None:
                    log(f"[nav] t={t:6.2f}s route to {goal} found only with margin {infl:.3f} m"
                        + (" using confirmed walls only" if conf else ""), echo=False)
                    break
            planner.build(clear_disc=(gx_, gy_, P["pillar_radius"] + P["pillar_clear"]))
        if path is None:
            path = explore(rc)
        if path is None:
            plan_fail += 1
            follow.set_path(None)                 # never keep driving along a stale path
            if plan_fail == 1 and DEBUG:
                save_debug(f"fail_{t:.2f}")
                rs = planner.free_start(rc)
                log(f"[dbg] plan failed: robot cell {rc} blocked={bool(planner.blocked[rc])} free_start={rs} "
                    f"goal={goal} known={goal in pillars} explore_target={explore_target}", echo=False)
            if plan_fail % 20 == 1:
                log(f"[nav] t={t:6.2f}s no path (goal {goal}) -> scanning on the spot")
        else:
            plan_fail = 0
            px, py = grid.xy(np.array([p[0] for p in path]), np.array([p[1] for p in path]))
            follow.set_path(np.c_[px, py])
    if DEBUG and t >= next_dbg:
        next_dbg += 3.0
        save_debug(f"t{int(t):03d}")
        print(f"[odom] t={t:6.2f} x={pose[0]:.3f} y={pose[1]:.3f} th={pose[2]:.3f}")
    # ---------------------------------------------------------- tilt reflex
    # driving forward into a wall corner or under a low slab can make the robot climb (nose up);
    # within a second it rears up to 40-50 deg and hangs there with the wheels off the floor.
    # At the first clear tilt while driving forward: back off at once, remember the obstacle
    if max(abs(pitch), abs(roll)) > P["tilt_reflex"] and cmd_v > 0.05 and t >= backup_until:
        stuck_count += 1
        stuck_where = (pose[0], pose[1])
        backup_until = t + P["backup_time"]
        log(f"[nav] t={t:6.2f}s tilting while driving (pitch {pitch:.2f}, roll {roll:.2f}) -> backing off, remembering obstacle ahead")
        yy = np.linspace(-0.13, 0.13, 11)
        for xx in (0.13, 0.16):
            wx, wy = to_world(pose, np.full_like(yy, xx), yy)
            grid.add_obstacle(wx, wy, weight=P["vis_hits"])
        last_plan_t = -1.0
    # ---------------------------------------------------------- stuck / back-up
    if t < backup_until:
        mode = "backing-up"
        # escalating escape manoeuvres when stuck repeatedly at the same place (wedged in a gap)
        v_b, w_b = RECOVERY[(stuck_count - 1) % len(RECOVERY)]
        boost = 1.0 if stuck_count < 3 else P["backup_boost"]      # wedged: pull out harder
        drive(v_b * P["backup_speed"] * boost, w_b * P["backup_turn"])
        return
    if t - progress_ref[0] > P["stuck_time"]:
        moved = math.hypot(pose[0] - progress_ref[1], pose[1] - progress_ref[2])
        turned = abs(wrap(pose[2] - progress_ref[3]))
        if math.hypot(pose[0] - stuck_where[0], pose[1] - stuck_where[1]) > P["stuck_reset_dist"]:
            stuck_count = 0
        if moved < 0.04 and turned < 0.15 and cmd_activity > 0.3 * P["stuck_time"] / DT \
                and gi < len(goals) and goals[gi] in pillars and goals[gi] in last_ranged:
            # blocked right next to the goal pillar (a low board or slab beside it stops the robot
            # before the 'reached' ring): if the pillar was ranged at its estimated place shortly
            # before, it counts as reached - as close as the surroundings allow
            g_ = goals[gi]; pg_ = pillars[g_]; t0_, ox_, oy_ = last_ranged[g_]
            d_ = math.hypot(pg_[1] - pose[0], pg_[2] - pose[1])
            if (d_ < P["reach_dist"] + P["reach_slack"] and t - t0_ < P["stuck_confirm_recent"]
                    and math.hypot(ox_ - pg_[1], oy_ - pg_[2]) < P["pillar_cluster"]):
                goal_reached(g_, t, f" (blocked next to it: {d_:.2f} m)")
                progress_ref = (t, pose[0], pose[1], pose[2])
                cmd_activity = 0
                return
        if moved < 0.04 and turned < 0.15 and cmd_activity > 0.3 * P["stuck_time"] / DT:
            stuck_count += 1
            stuck_where = (pose[0], pose[1])
            log(f"[nav] t={t:6.2f}s stuck ({stuck_count}x here) -> escape manoeuvre, remembering obstacle ahead")
            backup_until = t + P["backup_time"] * (1.0 + 0.5 * min(stuck_count - 1, 3))
            yy = np.linspace(-0.13, 0.13, 11)
            for xx in (0.13, 0.16):
                wx, wy = to_world(pose, np.full_like(yy, xx), yy)
                grid.add_obstacle(wx, wy, weight=P["vis_hits"])
            last_plan_t = -1.0
        progress_ref = (t, pose[0], pose[1], pose[2])
        cmd_activity = 0
    # ---------------------------------------------------------- control
    r_, c_ = grid.cell(pose[0], pose[1])
    clearance = float(planner.d_obst[r_, c_]) if planner.d_obst is not None else 1.0
    gd = math.hypot(*(follow.path[-1] - np.array(pose[:2]))) if follow.path is not None else 0.0
    v_cap = P["v_unverified"] if follow.unverified_ahead(grid, P["verify_dist"], P["verify_near"]) else None
    if gi < len(goals) and goals[gi] in pillars:
        pg = pillars[goals[gi]]
        if math.hypot(pg[1] - pose[0], pg[2] - pose[1]) < P["goal_slow_dist"]:
            v_cap = P["v_goal_near"] if v_cap is None else min(v_cap, P["v_goal_near"])
    v, w = follow.command(pose, DT, clearance, gd, v_cap)
    mode = "following-path" if follow.path is not None else "waiting-for-path"
    if follow.path is None and plan_fail > 2:
        mode = "no-path-scanning"
        v, w = 0.0, P["scan_rate"]                 # no plan: turn on the spot to gather information
    if abs(v) > 0.05 or abs(w) > 0.3:
        cmd_activity += 1
    drive(v, w)


def status_line():
    d_img = per.depth_array(depth.getRangeImage(data_type="buffer"))
    fin = np.isfinite(d_img)
    r_, c_ = grid.cell(pose[0], pose[1])
    blk = bool(planner.blocked[r_, c_]) if getattr(planner, "blocked", None) is not None else None
    return (f"[status] t={t:6.2f} mode={mode} goal={goals[gi] if gi < len(goals) else 'done'} "
            f"pose=({pose[0]:.3f},{pose[1]:.3f},{math.degrees(pose[2]):.0f}deg) yaw={pose[2]:.5f} "
            f"pillars={ {k: (round(v[1], 2), round(v[2], 2)) for k, v in pillars.items()} } "
            f"path={'none' if follow.path is None else len(follow.path)} plan_fail={plan_fail} robot_cell_blocked={blk} "
            f"lidar_valid={int(np.isfinite(lid_full).sum())} depth_valid={fin.mean():.2f} "
            f"depth_range=({np.nanmin(np.where(fin, d_img, np.nan)) if fin.any() else float('nan'):.2f},"
            f"{np.nanmax(np.where(fin, d_img, np.nan)) if fin.any() else float('nan'):.2f}) "
            f"wide_matches={matcher.wide_used} slip_fix={slipper.used} near_hazard={near_hazard_n[0]} wall_cells={int((grid.logodds > P['l_occ']).sum())} cam_obst={int(grid.vis_obst.sum())} green={int(grid.green.sum())}")


def snapshot():
    import cv2
    img = np.frombuffer(cam.getImage(), np.uint8).reshape(cam.getHeight(), cam.getWidth(), 4)[:, :, :3]
    cv2.imwrite(os.path.join(LOG_DIR, f"{WORLD}_camera.png"), img)
    d_img = per.depth_array(depth.getRangeImage(data_type="buffer")).copy()
    d8 = np.nan_to_num(np.clip(d_img, 0, 4) / 4 * 255, posinf=255).astype(np.uint8)
    cv2.imwrite(os.path.join(LOG_DIR, f"{WORLD}_depth.png"), d8)
    hsv = per.hsv(cam.getImage())
    log(f"[snap] saved camera/depth images; blue px {int(((hsv[:,:,0]>105)&(hsv[:,:,0]<135)&(hsv[:,:,1]>170)).sum())}, "
        f"yellow px {int(((hsv[:,:,0]>20)&(hsv[:,:,0]<38)&(hsv[:,:,1]>170)).sum())}, "
        f"green px {int(((hsv[:,:,0]>45)&(hsv[:,:,0]<75)&(hsv[:,:,1]>150)).sum())}, "
        f"centre pixel BGR {img[240, 320].tolist()} HSV {hsv[240, 320].tolist()}")


t = 0.0
lid_full = np.zeros(1)
lid_now = np.zeros(1)
crashed = False
while robot.step(DT_MS) != -1:
    if crashed:
        continue
    try:
        step_once()
        if t >= next_status:
            next_status += 1.0
            log(status_line(), echo=False)
        if not snap_done and t > 0.5:
            snap_done = True
            snapshot()
    except Exception:
        crashed = True
        drive(0.0, 0.0)                    # never leave the wheels turning after an error
        log("[nav] CONTROLLER ERROR - robot stopped:\n" + traceback.format_exc())


# Reference: the code in this file was generated with Claude (Anthropic LLM, claude.ai), Oct 2026,
# in a guided session for the Autonomous Robots Modularbeit (prompt: "implement navigation for maze 1").
