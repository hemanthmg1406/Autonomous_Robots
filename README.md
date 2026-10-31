# Rosbot navigation – Autonomous Robots Modularbeit

The robot drives start → blue pillar → yellow pillar using only its own sensors. There's no Supervisor and no map knowledge from the world files.

## Folder layout
```
Autonomous robots/
  navigator/                 the ONE shared codebase (used by every maze)
    navigator.py             main loop: sense -> map -> decide -> plan -> act
    params.py                all tunable values; optional per-maze overrides
    perception.py            lidar angles, green floor, depth obstacles, pillar detection
    grid.py                  occupancy grid (lidar walls, camera obstacles, green, camera coverage)
    lidar_deskew.py          corrects the rotating lidar (12 Hz) for robot motion
    localize.py              scan matching against the map (bilinear distance field, coarse-to-fine search)
    planner.py               cost map + minimum-cost path (scikit-image MCP)
    control.py               pure-pursuit path follower, speed limits
    map_view.py              live map window (occupancy grid, path, pillars) while the controller runs
  Mazes/MazeN/controllers/navigator/navigator.py   launcher: checks the Python, then starts ../../../../navigator
  tools/run_webots.sh        starts Webots R2025a (macOS 27 fix + project Python)
  tools/run_maze1..5.command double-click starters for Maze 1 to 5 (record console output)
  REFERENCES.md              published methods behind each part of the code
  docs/HOW_IT_WORKS.md       study guide: how every part works, with exercises
```

## Setup (macOS, once)
```bash
"$(pyenv which python3)" -m venv ~/.venvs/webots        # Python 3.10-3.13
source ~/.venvs/webots/bin/activate
pip install numpy opencv-python scipy scikit-image matplotlib
```

## Run
Any OS: open `Mazes/MazeN/worlds/MazeN.wbt` in Webots R2025a and press ▶ (the worlds already use the controller `navigator`; Python needs numpy, opencv-python, scipy and scikit-image).

On the Mac used for development, easiest: double-click **`tools/run_maze1.command`** in Finder. It starts Webots with Maze 1 and copies
all console output into `tools/last_run_output.txt`. Or from Terminal:
```bash
cd ~/Documents/"Autonomous robots"/tools
./run_webots.sh ../Mazes/Maze1/worlds/Maze1.wbt
```
Every run also writes `navigator/logs/<World>_<date>.log`: a status line per second (mode, pose,
pillars, planning state), a camera and depth snapshot, and any error with a full traceback. If an
error happens, the wheels are stopped.
`navigator/logs/launcher.log` records which Python started the controller. If that Python lacks the
packages, the launcher restarts itself with `~/.venvs/webots` automatically.
Press ▶. The console prints when each pillar is located and reached, and at the end a `RESULT` block with the times start -> blue pillar, blue -> yellow pillar and total (simulation time). All details are written to the log file.

While the controller runs, a window "Occupancy grid - MazeN" shows the map as the robot builds it (lidar
walls, camera obstacles, green floor, the no-go zone around obstacles, the planned path and the pillars,
and while a pillar is still unknown the exploration frontier, pillar candidates and the chosen target;
legend in `docs/HOW_IT_WORKS.md`, section 12). It is display only and does not change the robot's behaviour.
To hide it, start with `NAV_SHOW_MAP=0 ./run_webots.sh ...` or set `"show_map": false` in a parameter file.

To save map snapshots every 3 s (for debugging), start with:
`NAV_DEBUG=/tmp/navdbg ./run_webots.sh ../Mazes/Maze1/worlds/Maze1.wbt`. Create the folder first.

## Parameters
All values are in `navigator/params.py`. A maze may override *values only* through `navigator/params_<WorldName>.json`, for example `params_Maze1.json` containing `{"v_max": 0.9}`. Unknown keys are rejected. No file means defaults. Maze 1 uses the defaults.

## "Reached"
A pillar counts as reached when the robot centre is within 0.30 m of the pillar centre (robot front about 10 cm from the pillar) and the pillar was just seen there with a range sensor. If walls, green floor or a low board don't allow that, within 0.36 m.

## Approach (short)
1. **Odometry:** IMU heading plus the mean wheel-encoder travel, corrected by lidar **scan matching** against the map built so far.
2. **Lidar de-skew:** the RpLidarA2 is a *rotating* lidar (one sweep = 83 ms ≈ 2.6 control steps). Each beam is paired with the robot pose at the moment it was measured.
3. **Mapping layers:**
   - lidar walls, as a log-odds grid in which confirmed walls are "sticky";
   - **depth camera** points between 3 cm and 23.5 cm height, which catches *floating* walls (invisible to the lidar plane at 13 cm) and *low* boards; anything higher is ignored, so the robot drives under overhead walls;
   - front **infra-red** sensors, which cover the lidar's 20 cm blind zone;
   - **green floor** from the RGB camera, projected onto the floor plane;
   - **camera coverage**, i.e. where the depth camera has checked the floor.
4. **Pillars:** colour segmentation (HSV). The distance comes from the depth image, or from the lidar when the pillar is closer than 0.6 m. Estimates are fused over time.
5. **Planning:** minimum-cost path on a 2 cm grid.
   - Lidar walls are reduced to their centre line (all walls are 5 cm boards), so a few cm of localisation error can't close narrow passages.
   - The robot centre keeps 13 cm from walls and 15.5 cm from green, plus a soft cost that keeps it centred.
   - If no route exists, slightly smaller margins (12.5 cm, then 12 cm) are tried, finally with confirmed walls only.
   - Where the robot has already driven always counts as passable, so the way back can't close.
   - Unknown space is allowed but more expensive.
   - If the goal pillar is unknown, the robot turns once on the spot. It then explores towards space the camera hasn't checked, commits to each target, and looks around on arrival.
   - Camera obstacles are only added away from lidar walls and must form clusters (no specks).
6. **Control:** pure pursuit.
   - Speed is limited by wall clearance, by heading error, and to 0.35 m/s while the path 0.6–1.3 m ahead has not yet been checked by the depth camera (needed to see floating walls in time).
   - When the robot gets stuck it backs up and remembers an obstacle at that spot.
   - Tilt reflex: if the robot starts to tilt while driving forward (climbing a wall corner or wedging under a slab), it backs off at once and remembers the obstacle.
7. **Robustness additions:**
   - Wheel slip: the current lidar scan is also aligned with a scan from about a second earlier (independent of the map); if the scans show less motion than the wheels, the pose is corrected. While strongly tilted, wheel travel is ignored.
   - Slabs closer than the depth camera's 0.6 m minimum range: wooden board colour in the RGB image at robot-top height, where the depth image has no value and the lidar sees nothing nearby, is marked as an obstacle 0.5-0.6 m ahead; the mark is removed when the depth camera later sees that height free.
   - A pillar only counts as visible if no wall lies between robot and pillar (walls right next to the pillar included).
   - Exploration keeps the driving direction: a target straight ahead is preferred over one that needs turning off sideways or back.

## Results (Webots R2025a on macOS, simulation time)
Same code and the same default parameters for all five mazes (no per-maze parameter files, no per-maze switches).
Original start, times printed by the controller in its `RESULT` block:
| Maze | start -> blue pillar | blue -> yellow pillar | total |
|---|---|---|---|
| 1 | 39.49 s | 20.61 s | 60.10 s |
| 2 | 40.96 s | 14.27 s | 55.23 s |
| 3 | 17.86 s | 67.55 s | 85.41 s |
| 4 | 25.15 s | 20.54 s | 45.70 s |
| 5 | 7.94 s | 9.95 s | 17.89 s |

## References
The code was generated with Claude (Anthropic LLM, claude.ai), Oct 2026. Each file states this after its code, as required by the citation rules.
Libraries: NumPy, SciPy, OpenCV, scikit-image (MCP_Geometric for path search, skeletonize), Webots R2025a controller API.
Background:
- Frontier-based exploration: B. Yamauchi, "A frontier-based approach for autonomous exploration", IEEE CIRA 1997.
- Pure pursuit: R. C. Coulter, "Implementation of the Pure Pursuit Path Tracking Algorithm", CMU-RI-TR-92-01, 1992.
- Occupancy grids and likelihood fields for scan matching: S. Thrun, W. Burgard, D. Fox, "Probabilistic Robotics", MIT Press 2005.
