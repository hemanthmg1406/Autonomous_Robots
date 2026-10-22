"""De-skewing for the Rosbot's *rotating* RpLidarA2 (12 Hz, 400 beams, PROTO type "rotating").

Within one 32 ms control step the lidar head only sweeps ~154 beams, so one range image
contains beams measured up to 83 ms apart. Measured in Webots (calibration run): the head
advances in increasing beam index at freq * N beams/s. We estimate the head phase online
from which beams changed since the last step, compute each beam's measurement time, and
attach the robot pose interpolated at that time to each beam."""
import math
from collections import deque
import numpy as np


class LidarDeskew:
    def __init__(self, n, freq):
        self.n, self.rate = n, freq * n          # beams per second
        self.prev = None
        self.c_vec = 0j                          # circular mean of the phase offset
        self.hist = deque(maxlen=8)              # (t, x, y, yaw_unwrapped)

    def add_pose(self, t, x, y, yaw):
        if self.hist:
            yaw = self.hist[-1][3] + math.remainder(yaw - self.hist[-1][3], 2 * math.pi)
        self.hist.append((t, x, y, yaw))

    def _observe_phase(self, t, cur):
        if self.prev is None:
            return
        a = np.where(np.isfinite(cur), cur, -1.0)
        b = np.where(np.isfinite(self.prev), self.prev, -1.0)
        ch = np.nonzero(np.abs(a - b) > 1e-6)[0]
        if len(ch) < 40:
            return
        # the changed beams form one arc; its end is just before the largest gap (circularly)
        s = np.sort(ch)
        gaps = np.diff(np.r_[s, s[0] + self.n])
        end = s[int(np.argmax(gaps))]
        c = (end + 0.5 - self.rate * t) % self.n
        self.c_vec += np.exp(2j * math.pi * c / self.n)

    def beam_poses(self, t, ranges):
        """returns per-beam (x, y, yaw) arrays of the robot pose when each beam was measured"""
        cur = np.asarray(ranges, float)
        self._observe_phase(t, cur)
        self.prev = cur
        h = np.array(self.hist)
        if abs(self.c_vec) < 1e-9 or len(h) < 2:
            age = np.zeros(self.n)
        else:
            c = (math.atan2(self.c_vec.imag, self.c_vec.real) / (2 * math.pi) * self.n) % self.n
            head = (self.rate * t + c) % self.n
            age = ((head - np.arange(self.n) - 0.5) % self.n) / self.rate
        tb = np.clip(t - age, h[0, 0], h[-1, 0])
        return np.interp(tb, h[:, 0], h[:, 1]), np.interp(tb, h[:, 0], h[:, 2]), np.interp(tb, h[:, 0], h[:, 3])


# Reference: the code in this file was generated with Claude (Anthropic LLM, claude.ai), Oct 2026,
# in a guided session for the Autonomous Robots Modularbeit (prompt: "implement navigation for maze 1").
