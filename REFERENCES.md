# References

## Authorship of the code
The code text in `navigator/*.py` was generated with Claude (Anthropic LLM, claude.ai, Oct 2026; `map_view.py`
with Claude Code, Oct 2026), in an
iterative session in which the group set the task, tested the robot in Webots, analysed failures
and decided on changes. Each file says this in its last lines. As far as we know, no code was copied
from GitHub, Stack Overflow or other websites.

The code implements well-known, published methods. The table below lists, for each part of the code,
the method and the publication it comes from, so that the reader can find the original idea quickly.
All sources were published before 2025.

## Methods used, by file

| File / function | Method | Source |
|---|---|---|
| `grid.py` `Grid.update_lidar` | Occupancy grid with log-odds update: free cells along each beam, a hit at its end | H. Moravec, A. Elfes, "High resolution maps from wide angle sonar", Proc. IEEE ICRA 1985, pp. 116-121. S. Thrun, W. Burgard, D. Fox, *Probabilistic Robotics*, MIT Press 2005, ch. 9 (occupancy grid mapping, log-odds form) |
| `grid.py` `Grid.obstacle_distance` | Obstacle distance map by Euclidean distance transform; wall bands thinned to their centre line by skeletonization | Distance transform: C. R. Maurer, R. Qi, V. Raghavan, "A linear time algorithm for computing exact Euclidean distance transforms of binary images in arbitrary dimensions", IEEE TPAMI 25(2), 2003 (algorithm behind `scipy.ndimage.distance_transform_edt`). Skeleton: T. Y. Zhang, C. Y. Suen, "A fast parallel algorithm for thinning digital patterns", Comm. ACM 27(3), 1984 (algorithm behind `skimage.morphology.skeletonize`) |
| `grid.py` `Grid.vis_obst`, `pillar_candidates` | Connected-component labelling of map cells | `scipy.ndimage.label`, SciPy reference documentation (P. Virtanen et al., "SciPy 1.0", Nature Methods 17, 2020) |
| `localize.py` `ScanMatcher` | Scan matching against a likelihood field (distance to the nearest mapped wall), coarse-to-fine correlative search over x/y | Thrun, Burgard, Fox, *Probabilistic Robotics*, 2005, ch. 6.4 (likelihood field range model). E. Olson, "Real-time correlative scan matching", Proc. IEEE ICRA 2009 (coarse-to-fine correlative search) |
| `localize.py` `SlipCheck` | Scan-to-scan alignment against an earlier key-frame scan, used to detect wheel slip | Same likelihood-field / correlative matching as above, applied scan-to-scan; key frames as in F. Lu, E. Milios, "Robot pose estimation in unknown environments by matching 2D range scans", J. Intelligent and Robotic Systems 18, 1997 |
| `lidar_deskew.py` | Motion compensation (de-skewing) of a rotating lidar: each beam gets the robot pose interpolated at its measurement time | J. Zhang, S. Singh, "LOAM: Lidar Odometry and Mapping in Real-time", Robotics: Science and Systems 2014 (motion distortion compensation) |
| `navigator.py` odometry | Differential/skid-steer dead reckoning: wheel encoder travel and IMU heading | R. Siegwart, I. Nourbakhsh, D. Scaramuzza, *Introduction to Autonomous Mobile Robots*, 2nd ed., MIT Press 2011, ch. 3 and 5 |
| `perception.py` `pillar`, `green_points`, `near_hazard` | Colour segmentation in HSV space; connected components of the colour mask | G. Bradski, "The OpenCV Library", Dr. Dobb's Journal 2000; OpenCV documentation for `cv2.cvtColor` (HSV) and `cv2.connectedComponentsWithStats` |
| `perception.py` `green_points`, `obstacle_points`, `band_clear_points` | Pinhole camera model: back-projection of depth / floor pixels to 3D points in the robot frame | R. Hartley, A. Zisserman, *Multiple View Geometry in Computer Vision*, 2nd ed., Cambridge University Press 2004, ch. 6 |
| `planner.py` `Planner.plan` | Minimum-cost path on a grid (Dijkstra) | E. W. Dijkstra, "A note on two problems in connexion with graphs", Numerische Mathematik 1, 1959. Implementation: `skimage.graph.MCP_Geometric`, S. van der Walt et al., "scikit-image: image processing in Python", PeerJ 2:e453, 2014 |
| `planner.py` `Planner.build` | Cost map with obstacle inflation by the robot radius and a soft cost near obstacles | D. V. Lu, D. Hershberger, W. D. Smart, "Layered costmaps for context-sensitive navigation", Proc. IEEE/RSJ IROS 2014 (inflation layer of ROS `costmap_2d`) |
| `planner.py` `frontier_mask`, `navigator.py` `_explore` | Frontier-based exploration: drive to the border between known free and unknown space | B. Yamauchi, "A frontier-based approach for autonomous exploration", Proc. IEEE CIRA 1997 |
| `control.py` `Follower.command` | Pure pursuit path tracking (curvature 2·sin(α)/L to a look-ahead point) | R. C. Coulter, "Implementation of the Pure Pursuit Path Tracking Algorithm", Tech. Rep. CMU-RI-TR-92-01, Carnegie Mellon University, 1992 |
| `control.py` `wheel_speeds` | Differential-drive inverse kinematics (v, ω → left/right wheel speed) | Siegwart, Nourbakhsh, Scaramuzza 2011, ch. 3 |
| `map_view.py` `MapView` | Display of the occupancy grid as probabilities p = 1/(1+e^-l) from the log-odds; drawing with OpenCV | Thrun, Burgard, Fox, *Probabilistic Robotics*, 2005, ch. 9 (log-odds to probability); OpenCV documentation for `cv2.imshow`, `cv2.polylines`, `cv2.circle`, `cv2.putText` |
| all files | Webots controller API (Robot, Lidar, Camera, RangeFinder, InertialUnit, PositionSensor, DistanceSensor, Motor) | Cyberbotics, *Webots Reference Manual*, R2025a; O. Michel, "Webots: Professional Mobile Robot Simulation", Int. J. Advanced Robotic Systems 1(1), 2004 |

## Libraries
NumPy, SciPy, OpenCV (cv2), scikit-image, Matplotlib (debug images only), Webots R2025a controller API.
