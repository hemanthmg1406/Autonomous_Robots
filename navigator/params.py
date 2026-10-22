"""All tunable numbers in one place. The same code runs in every maze; a maze may only
override values here (via params_<WorldName>.json next to this file), never switch code paths."""
import json, os

DEFAULT = {
    # --- robot / motion ---------------------------------------------------------
    "wheel_radius": 0.043,
    "track_eff": 0.36,          # effective track width of the skid-steer base [m]
    "wheel_max": 25.9,          # rad/s (Webots rejects exactly the motor limit of 26)
    "v_max": 1.1,               # m/s cruise speed
    "v_min": 0.12,              # m/s minimum forward speed while following a path
    "w_max": 4.0,               # rad/s max turn rate
    "turn_in_place_angle": 0.9, # rad: heading error above which the robot turns on the spot
    "lookahead": 0.35,          # m, pure-pursuit look-ahead distance (in open space)
    "lookahead_min": 0.15,      # m, look-ahead close to walls
    "accel": 2.5,               # m/s^2 speed ramp
    "decel": 6.0,               # m/s^2 limit for any speed change sent to the wheels
    "w_accel": 20.0,            # rad/s^2 limit for any turn-rate change sent to the wheels
    "slow_clearance": 0.15,     # m of extra clearance (beyond inflate_obst) needed for full speed
    "slow_min_factor": 0.5,    # speed factor when the robot is as close to a wall as allowed
    "v_unverified": 0.35,       # m/s while the path ahead has not been checked by the depth camera
    "v_goal_near": 0.4,         # m/s within goal_slow_dist of a known pillar (so the camera can confirm it)
    "goal_slow_dist": 0.8,      # m
    "v_tight": 0.25,            # m/s in narrow gaps ...
    "tight_clearance": 0.14,    # m ... where the robot centre is closer than this to a wall
    "verify_dist": 1.3,         # m: path from verify_near to here must be camera-checked for full speed
    "verify_near": 0.6,         # m, depth camera minimum range
    "shadow_dist": 0.3,         # m, unverified cells this far behind a seen floating edge count as obstacle
    "shadow_wall_gap": 0.15,    # m, camera obstacles closer than this to a lidar wall cast no shadow
    "band_lo": 0.18,            # m, height band that must be seen free before driving fast:
    "band_hi": 0.21,            #    above the lidar plane, below the top of the camera housing
    "band_max": 3.0,            # m, rays are followed this far
    # --- map --------------------------------------------------------------------
    "res": 0.02,                # m per grid cell
    "size": 16.0,               # m, square map centred on the start position
    "map_max_turn_rate": 1.0,   # rad/s, lidar walls are only mapped while turning slower than this
    "lidar_height": 0.131,
    "lidar_min_range": 0.2,     # m, shorter readings (0 before the first sweep) are invalid      # m, lidar plane above the floor
    "floor_hit_share": 0.7,     # returns beyond this share of the tilted beam's floor distance are dropped
    "map_max_tilt": 0.04,       # rad, no lidar mapping while the robot is tilted more than this
    "open_ray_from": 0.3,       # m, beams without return count as 'seen empty' from here to lidar_max_use
    "lidar_max_use": 5.0,       # m, ignore lidar returns further away
    "l_hit": 0.9, "l_free": -0.35, "l_min": -3.0, "l_max": 5.0, "l_occ": 0.7,
    "wall_close_gap": 0.12,     # m, gaps between mapped walls narrower than this are closed
    "wall_half": 0.025,         # m, half the thickness of the maze boards (all walls are 5 cm boards)
    "local_refresh": False,     # near the robot, wall cells the latest scans saw through are not used for planning
    "local_radius": 1.5,        # m
    "local_scans": 6,
    "local_free_stop": 0.03,    # m before a hit that the latest rays count as having passed           # number of latest scans used (~0.4 s)
    "sticky_clear": 0.1,        # free-ray evidence factor for cells above l_sticky
    "l_sticky": 1.5,            # log-odds above which a wall cell is hardly cleared any more
    "free_stop": 0.06,          # m, free-space clearing stops this far before a lidar hit
    # --- obstacles seen by the depth camera (floating / low walls) -------------
    "cam_height": 0.198,        # m, measured optical centre height
    "obst_z_min": 0.03,         # m, points higher than this are not floor
    "float_z_min": 0.14,        # m, obstacle points above this (lidar plane 0.13) with nothing below are 'floating'
    "float_same_range": 0.08,   # m, a point lower in the image column this close in range = the object stands on the floor
    "obst_z_max": 0.243,        # m, points lower than this would hit the robot (robot top 0.214)
    "depth_max_use": 3.0,       # m
    "vis_hits": 2,              # camera detections needed before a cell counts as obstacle
    "vis_min_cluster": 4,       # camera obstacle cells must form a cluster at least this big
    "cam_lidar_gap": 2,         # cells: camera points this close to a lidar wall are not added
    "cam_max_turn_rate": 2.0,   # rad/s, camera images taken while turning faster are not mapped
    "cam_max_tilt": 0.03,       # rad, depth images taken while the robot pitches/rolls more are not mapped
    # --- green floor --------------------------------------------------------------
    "green_max_dist": 3.0,      # m, only trust ground projection up to here
    "green_plus": 2,            # evidence per green sighting; a cell is green at >= this
    "green_ev_min": -2, "green_ev_max": 6,   # evidence limits (plain floor counts -1)
    # --- planning -----------------------------------------------------------------
    "inflate_obst": 0.13,       # m, robot centre must stay this far from walls (half width 0.116)
    "inflate_green": 0.175,     # m, ... and this far from green (corners when turning: 0.153)
    "fallback_confident": False,   # last resort: plan with confirmed (sticky) walls only - opens false gaps
    "fallback_inflation": [0.125, 0.12],  # smaller margins tried (in order) when no route exists
    "soft_dist": 0.35,          # m, extra cost when closer than this to an obstacle
    "soft_cost": 15.0,
    "unverified_cost": 2.0,     # cost factor where the camera-housing height band is not yet seen free
    "unknown_cost": 1.3,        # cost factor for driving through unexplored cells
    "replan_period": 0.25,      # s
    "sweep_margin": 0.03,       # m, robot footprint shrunk by this before clearing the map under it
    "trail_cost": 10.0,         # extra cost for using an already driven cell that looks blocked now
    # --- goals -----------------------------------------------------------------------
    "pillar_radius": 0.10,
    "pillar_obs_keep": 40,      # observations kept per pillar colour
    "pillar_cluster": 0.15,     # m, observations closer than this support each other
    "sight_full_dist": 1.5,     # m, sightings closer than this count fully when fusing the pillar position
    "pillar_min_obs": 3,        # agreeing observations needed before a pillar position is used
    "pillar_tight": 0.04,       # m, ... or 2 sightings this close together
    "pillar_agree": 0.12,       # m, depth camera and lidar distance must agree this well
    "pillar_depth_iqr": 0.08,   # m, larger spread of depth values inside the blob = partly hidden
    "pillar_width_tol": 0.25,   # relative tolerance between width-based and measured distance
    "pillar_sliver": 1.5,       # apparent width this much narrower than the distance implies -> rejected
    "pillar_min_height": 0.3,   # m, pillars are at least this tall
    "border_lidar_max": 0.7,
    "near_width_max": 0.45,     # m, closer than this (both range sensors blind) the apparent width is used    # m, a pillar cut by the image border is located by the lidar alone up to here
    "face_time": 1.5,
    "approach_keep_support": 2, # estimates with at least this many sightings survive a failed approach ...
    "approach_tries": 3,        # ... this many times (then discarded)
    "approach_exclude": 0.35,   # m around a failed approach point excluded from the reach area
    "confirm_min_dist": 0.36,   # m, closer than this an unconfirmed estimate is checked again from further back
    "backoff_time": 0.8,        # s backing away for that ...
    "backoff_wait": 4.0,        # s ... at most once per this time           # s, turning towards a pillar estimate that is out of the camera view
    "confirm_recent": 3.0,      # s, "reached" needs a sighting with a range-sensor distance this recent at the estimate
    "reach_dist": 0.30,         # m, robot centre to pillar centre counts as "reached" (front ~12 cm from pillar)
    "reach_slack": 0.06,        # m, if the pillar cannot be approached closer, this much extra is accepted
    "pillar_clear": 0.12,       # m around the pillar surface removed from the obstacle map when it is the goal (the robot centre may not enter this disc)
    "pillar_los_clear": 0.06,   # m around the pillar surface ignored in line-of-sight tests (walls right next to the pillar still block the view)
    "frontier_min_dist": 0.4,   # m
    "explore_box_margin": 0.6,  # m, frontiers beyond the confirmed walls' bounding box + this are ignored
    "explore_box_min_cells": 200,  # wall cells needed before that box is used
    "explore_reached": 0.3,     # m, an exploration target closer than this counts as visited
    "spin_look_radius": 2.0,    # m, area considered when deciding whether a look-around is worth it
    "spin_min_unchecked": 0.3,  # look around only if more than this share of it is camera-unchecked
    "frontier_structure_w": 1.5,   # m of extra path cost for a frontier with no walls around it
    "frontier_structure_r": 1.0,   # m, neighbourhood for that wall density
    "explore_lidar_frontier": False,  # explore space the lidar has not seen before camera-unchecked floor
    "frontier_gain_w": 1.5,     # m of path a frontier is worth if all space around it is lidar-unseen
    "frontier_gain_r": 0.6,     # m, neighbourhood for that
    "narrow_penalty": 3.0,      # extra selection cost per cell (2 cm) only passable with the smallest margin
    "explore_forward_w": 1.0,   # m of extra cost for an exploration target straight behind the robot (less to the side)
    "explore_persist": 0.5,     # m of extra cost per m a new frontier is away from the previous target
    "explore_soft_share": 0.1,  # share of the centring cost used when choosing the nearest frontier
    "blob_bonus": 0.7,          # m of path a pillar-candidate viewpoint is worth over a plain frontier
    "blob_max": 0.35,           # m, lidar objects at most this big may be a pillar (r = 0.1 m)
    "blob_min_cells": 8,
    "blob_min_width": 0.018,    # m, std of the cells across the object (wall fragments are thinner)
    "blob_min_ratio": 0.3,      # across/along spread: arcs are round-ish, wall fragments elongated
    "blob_view_min": 0.6,       # m, viewpoint distance range for looking at a pillar candidate
    "blob_view_max": 1.5,
    "look_min_range": 0.6,      # m, closer than this a pillar is not reliably recognised (fills the image)
    "look_range": 2.5,          # m, lidar hits closer than this inside the camera view count as looked at
    "look_half_angle": 0.3,     # rad, only hits near the image centre count as looked at (reliable detection)
    "look_fov_margin": 0.1,     # rad, image border strip where pillars are not trusted
    "visited_radius": 0.5,      # m around a reached exploration target that counts as checked
    "look_around_rate": 4.0,    # rad/s spin used to look for a pillar whose position is unknown
    "spin_clearance": 0.17,     # m to the nearest wall needed for that spin (robot corner radius 0.153)
    # --- localisation ------------------------------------------------------------
    "match_field_period": 0.5,  # s, how often the scan-matching distance field is rebuilt
    "match_every": 2,           # control steps between scan matches (and lidar map updates)
    "match_skeleton": False,    # match against the walls' centre lines (+ half a board) instead of all wall cells
    "match_cap": 0.10,          # m, distance field is cut off here (points further away are ignored)
    "match_range": 4.0,         # m, only lidar points closer than this are matched
    "match_max_shift": 0.06,    # m, larger corrections in one scan are rejected as implausible
    "slip_accept": 0.7,         # 'the wheels slipped' is accepted if it fits this much better than odometry
    "odo_max_tilt": 0.2,        # rad, encoder motion is ignored while the robot is tilted more than this
    "hz_dv_lo": -0.07,          # image rows (tan of elevation, down positive) for too-close slab edges: at 0.25-0.6 m
    "hz_dv_hi": 0.06,           # ... these rows are 0.16-0.24 m high (above the lidar plane, below the robot top)
    "wood_h_lo": 5,             # OpenCV hue range (0-180) and minimum saturation of the wooden boards
    "wood_h_hi": 24,
    "wood_s_min": 64,
    "hz_min_rows": 3,           # (sub-sampled) rows per column needed
    "hz_lidar_margin": 0.15,    # m, the lidar must see nothing closer than the depth camera's range (+ this) in that direction
    "hz_excl": 0.14,            # rad, ... within this angle around the column's bearing
    "hz_min_cols": 6,           # sub-sampled image columns needed for a slab edge
    "hz_clear_lo": 0.20,        # m, a slab mark is cleared where the depth camera saw this height band free
    "hz_clear_hi": 0.214,       # ...  up to the robot's top
    "hz_near": 0.5,             # m, cells along the ray between these distances are marked
    "hz_far": 0.6,
    "slip_every": 8,            # control steps between key-frame slip checks
    "slip_max_tilt": 0.08,      # rad, no key-frame check while tilted more than this
    "slip_range": 4.0,          # m, lidar points used for the key-frame check
    "slip_cap": 0.08,           # m, distance cut-off of the key-frame fit
    "slip_min_move": 0.06,      # m, odometry motion since the key frame before checking
    "slip_min_corr": 0.04,      # m, smaller corrections are left to the map matcher
    "slip_ratio": 0.5,          # correction only if it halves the misfit ...
    "slip_min_cost": 0.0003,    # m^2 ... and the odometry pose fits clearly badly
    "slip_max_corr": 0.2,       # m, larger single slip corrections are rejected
    "slip_tol": 0.03,           # m, tolerance of the slip plausibility test (corrected motion vs. wheel motion)
    "slip_map_ratio": 0.9,      # a slip correction must fit the map at least this much better than the odometry pose
    "slip_keyframe_t": 1.0,     # s, key frame age limit
    "slip_keyframe_d": 0.35,    # m, key frame renewed after this much motion
    "match_bad": 0.0012,        # m^2 mean squared distance above which the fit counts as poor ...
    "match_wide": 0.24,         # m ... and a wider search (+-) is tried
    # --- safety ------------------------------------------------------------------------
    "tilt_reflex": 0.15,        # rad, tilt while driving forward that triggers an immediate back-off
    "stuck_time": 2.0,          # s without progress -> back up
    "spin_stall_rate": 0.1,     # rad/s, a look-around spin slower than this for stuck_time is given up
    "stuck_confirm_recent": 6.0,  # s, a pillar ranged this recently counts as reached when the robot is stuck right next to it
    "ir_use_max": 1.0,          # m, front infra-red readings below this are mapped as obstacles
    "backup_speed": 0.25,       # m/s when backing out of a stuck situation
    "backup_time": 0.6,         # s
    "backup_boost": 3.0,        # speed factor for escapes after the 3rd failed attempt at one spot
    "backup_turn": 1.5,         # rad/s turn used in the escape manoeuvres
    "stuck_reset_dist": 0.3,    # m, moving this far from the last stuck place resets the escalation
    "scan_rate": 0.8,           # rad/s turn on the spot while no path is available (slower than map_max_turn_rate)
}

def load(world_path):
    p = dict(DEFAULT)
    name = os.path.splitext(os.path.basename(world_path or ""))[0]
    f = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"params_{name}.json")
    if os.path.exists(f):
        with open(f) as fh:
            over = json.load(fh)
        unknown = set(over) - set(DEFAULT)
        if unknown:
            raise KeyError(f"unknown parameter(s) in {f}: {unknown}")
        p.update(over)
    p["_param_file"] = f if os.path.exists(f) else "defaults"
    return p


# Reference: the code in this file was generated with Claude (Anthropic LLM, claude.ai), Oct 2026,
# in a guided session for the Autonomous Robots Modularbeit (prompt: "implement navigation for maze 1").
