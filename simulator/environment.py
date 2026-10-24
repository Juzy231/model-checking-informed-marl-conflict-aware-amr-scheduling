"""AMR gridworld simulator used for the reported controller evaluations."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math
import random
from typing import Iterable

Coord = tuple[int, int]
ACTION_HEADS = (3, 3, 3)  # speed gear, battery-threshold delta, obstacle sensitivity


@dataclass
class Robot:
    idx: int
    kind: str
    pos: Coord
    goal: Coord
    battery: float = 0.80
    threshold: float = 0.35
    gear: int = 1
    sensitivity: int = 1
    move_credit: float = 0.0
    charging: bool = False
    tasks_done: int = 0
    collisions: int = 0
    interventions: int = 0
    failure_step: int | None = None
    failed: bool = False
    history: deque = field(default_factory=lambda: deque(maxlen=10))
    phase: str = "Transfer"
    authorized: bool = True
    pause_reason: str = ""
    suspended_goal: Coord | None = None


class PaperAMREnv:
    """10x10 task-and-motion simulator with six heterogeneous robots."""
    def __init__(self, seed: int = 0, n_agents: int = 6, horizon: int = 500,
                 shield: bool = True, risk_penalty: bool = True,
                 p_loss: float = 0.005, p_fail: float = 0.005,
                 congestion: float = 0.25, p_robot_fail: float = 0.0,
                 rule_variant: str | None = None, auth_probability: float = 0.5,
                 r1_pause_threshold: float = 0.30,
                 r1_recovery_threshold: float = 0.30):
        self.seed = seed
        self.rng = random.Random(seed)
        self.n_agents = n_agents
        self.horizon = horizon
        self.shield = shield
        self.risk_penalty = risk_penalty
        self.p_loss = p_loss
        self.p_fail = p_fail
        self.congestion = congestion
        self.p_robot_fail = p_robot_fail
        if rule_variant not in (None, "R0_R1_R2", "R0_R1", "R0_R2", "R0_only"):
            raise ValueError("rule_variant must be None, 'R0_R1_R2', 'R0_R1', 'R0_R2', or 'R0_only'")
        if not 0.0 <= auth_probability <= 1.0:
            raise ValueError("auth_probability must be in [0, 1]")
        self.rule_variant = rule_variant
        self.r1_enabled = rule_variant in ("R0_R1", "R0_R1_R2")
        self.r2_enabled = rule_variant in ("R0_R2", "R0_R1_R2")
        self.auth_probability = auth_probability
        if not 0.0 < r1_pause_threshold < 1.0:
            raise ValueError("r1_pause_threshold must be in (0, 1)")
        self.r1_pause_threshold = r1_pause_threshold
        if not 0.0 <= r1_recovery_threshold <= 1.0:
            raise ValueError("r1_recovery_threshold must be in [0, 1]")
        self.r1_recovery_threshold = r1_recovery_threshold
        if any(p < 0 or p > 1 for p in (p_loss,p_fail,p_robot_fail)):
            raise ValueError('failure and packet-loss probabilities must be in [0,1]')
        self.width = self.height = 10
        self.chargers: tuple[Coord, ...] = ((0, 0), (9, 9))
        self.hazard: set[Coord] = {(4, 4), (4, 5), (5, 4), (5, 5)}
        self.walls: set[Coord] = {(2, 2), (2, 3), (2, 6), (2, 7),
                                  (7, 2), (7, 3), (7, 6), (7, 7)}
        self.kinds = ["heavy", "fast", "shuttle"]
        self.energy_per_move = {"heavy": 0.018, "fast": 0.014, "shuttle": 0.010}
        self.max_gear = {"heavy": 1, "fast": 2, "shuttle": 1}
        self.steps = 0
        self.robots: list[Robot] = []
        self._path_cache: dict[tuple[Coord, Coord], list[Coord]] = {}
        self.collision_events_total = 0
        self.r0_blocks_total = 0
        self.r1_pauses_total = 0
        self.r2_entries_total = 0
        self.pause_steps_total = 0
        self.local_steps_total = 0
        self.local_avoidances_total = 0
        self.reset()

    def free(self, p: Coord) -> bool:
        return 0 <= p[0] < self.height and 0 <= p[1] < self.width and p not in self.walls

    def neighbors(self, p: Coord) -> list[Coord]:
        r, c = p
        return [q for q in ((r-1,c),(r+1,c),(r,c-1),(r,c+1)) if self.free(q)]

    def path(self, start: Coord, goal: Coord) -> list[Coord]:
        key=(start,goal)
        if key in self._path_cache: return self._path_cache[key]
        if start == goal:
            return [start]
        q = deque([start]); parent = {start: None}
        while q:
            u = q.popleft()
            for v in self.neighbors(u):
                if v in parent: continue
                parent[v] = u
                if v == goal:
                    out = [v]
                    while out[-1] != start: out.append(parent[out[-1]])
                    result=list(reversed(out)); self._path_cache[key]=result; return result
                q.append(v)
        self._path_cache[key]=[start]
        return [start]

    def sample_goal(self, start: Coord) -> Coord:
        pool = [(r,c) for r in range(self.height) for c in range(self.width)
                if self.free((r,c)) and (r,c) not in self.chargers and (r,c) != start]
        # Congestion/task-hotspot control: with probability
        # `congestion`, sample a destination in the central aisle/intersection area.
        hot = [p for p in pool if 3 <= p[0] <= 6 and 3 <= p[1] <= 6]
        if hot and self.rng.random() < self.congestion: return self.rng.choice(hot)
        return self.rng.choice(pool)

    def reset(self) -> list[list[float]]:
        self.steps = 0
        self.collision_events_total = 0
        self.r0_blocks_total = 0
        self.r1_pauses_total = 0
        self.r2_entries_total = 0
        self.pause_steps_total = 0
        self.local_steps_total = 0
        self.local_avoidances_total = 0
        # Preserve the original nine start cells exactly. For scaling runs,
        # append deterministic staging cells spread across the aisles/perimeter.
        starts = [(0, 3), (0, 6), (9, 3), (9, 6), (3, 0), (6, 9), (0, 1), (9, 8), (5, 0)]
        occupied = set(starts)
        candidates = [(r, c) for r in (1, 3, 6, 8) for c in (1, 3, 6, 8)]
        candidates += [(r, c) for r in range(self.height) for c in range(self.width)
                       if r in {0, 1, 8, 9} or c in {0, 1, 8, 9}]
        for p in candidates:
            if (self.free(p) and p not in self.hazard and p not in self.chargers
                    and p not in occupied):
                starts.append(p)
                occupied.add(p)
        self.robots = []
        for i in range(self.n_agents):
            if i >= len(starts): raise ValueError(f"paper-grid implementation supports at most {len(starts)} agents")
            p = starts[i]
            kind = self.kinds[i % len(self.kinds)]
            rb = Robot(i, kind, p, self.sample_goal(p))
            if self.rule_variant is not None:
                # The selected-rule response sensitivity experiment samples one
                # authorization token per robot and episode.
                rb.authorized = self.rng.random() < self.auth_probability
            # Held-out single-robot fault: with probability p_robot_fail per
            # episode, the robot becomes immobile at a uniformly sampled tick.
            if self.p_robot_fail > 0 and self.rng.random() < self.p_robot_fail:
                rb.failure_step = self.rng.randint(1,self.horizon)
            rb.history.extend([1.0] * 5)
            self.robots.append(rb)
        return self.observations()

    def dist_charger(self, p: Coord) -> int:
        return min(abs(p[0]-c[0]) + abs(p[1]-c[1]) for c in self.chargers)

    def observations(self) -> list[list[float]]:
        out = []
        for a in self.robots:
            crowd = sum(1 for b in self.robots if b.idx != a.idx and
                        abs(a.pos[0]-b.pos[0]) + abs(a.pos[1]-b.pos[1]) <= 2)
            load = -1.0 if a.failed else 0.0 if a.charging else 1.0
            out.append([a.battery, load, self.dist_charger(a.pos)/18.0,
                        min(crowd/5.0, 1.0),
                        sum(a.history)/max(len(a.history),1),
                        a.gear/2.0, a.threshold, a.sensitivity/2.0])
        return out

    def _apply_action(self, robot: Robot, action: Iterable[int]) -> None:
        gear, threshold_delta, sensitivity = [int(x) for x in action]
        robot.gear = max(0, min(self.max_gear[robot.kind], gear))
        robot.threshold += (threshold_delta - 1) * 0.02
        if self.shield:
            robot.threshold = max(0.30, min(0.50, robot.threshold))
        else:
            robot.threshold = max(0.0, min(0.90, robot.threshold))
        robot.sensitivity = max(0, min(2, sensitivity))
        if robot.battery <= robot.threshold and not robot.charging:
            robot.charging = True
            robot.goal = min(self.chargers,
                             key=lambda c: abs(c[0]-robot.pos[0])+abs(c[1]-robot.pos[1]))

    def _proposal(self, robot: Robot, connected: bool) -> list[Coord]:
        if robot.failed:
            return [robot.pos]
        if self.rule_variant is not None and robot.phase == "Pause":
            if robot.pause_reason != "R1" or not robot.charging:
                return [robot.pos]
        if self.r2_enabled and robot.phase == "Local" and robot.pos in self.chargers:
            return [robot.pos]
        if self.rule_variant is not None and not self.r2_enabled and not connected and robot.phase in ("Transfer", "Complete"):
            # Without R2 the disconnected controller waits for central commands.
            return [robot.pos]
        if robot.charging and robot.pos in self.chargers:
            return [robot.pos]
        if not robot.charging and robot.pos == robot.goal:
            robot.tasks_done += 1
            robot.history.append(1.0)
            robot.goal = self.sample_goal(robot.pos)
        route = self.path(robot.pos, robot.goal)
        speed = robot.gear
        # Speed gears encode 0.5, 1.0, and 1.5 grid moves per simulation tick.
        # Fractional progress is carried between ticks; the local speed cap of 2
        # allows a high-gear robot to traverse two cells on alternating ticks.
        rate = (0.5, 1.0, 1.5)[speed]
        if robot.kind != "fast":
            rate = min(rate, 1.0)
        robot.move_credit += rate
        nmove = min(2, int(robot.move_credit))
        if not connected and robot.sensitivity >= 2:
            nmove = min(nmove, 1)
        if self.r2_enabled and robot.phase == "Local":
            nmove = min(nmove, 1)
        robot.move_credit -= nmove
        path = route[:nmove+1]
        return path if path else [robot.pos]

    @staticmethod
    def _conflict_pairs(paths: list[list[Coord]]) -> set[tuple[int,int]]:
        pairs: set[tuple[int,int]] = set()
        depth = max(map(len, paths), default=1)
        for t in range(1, depth):
            for i in range(len(paths)):
                pi = paths[i][min(t, len(paths[i])-1)]
                for j in range(i+1, len(paths)):
                    pj = paths[j][min(t, len(paths[j])-1)]
                    if pi == pj:
                        pairs.add((i,j))
                    pi0 = paths[i][min(t-1,len(paths[i])-1)]
                    pj0 = paths[j][min(t-1,len(paths[j])-1)]
                    if pi == pj0 and pj == pi0 and pi != pj:
                        pairs.add((i,j))
        return pairs

    def step(self, actions: list[Iterable[int]]):
        if len(actions) != self.n_agents:
            raise ValueError(f"expected {self.n_agents} actions, got {len(actions)}")
        self.steps += 1
        newly_failed=[]
        for rb in self.robots:
            if not rb.failed and rb.failure_step == self.steps:
                rb.failed=True;newly_failed.append(rb.idx)
        for rb, act in zip(self.robots, actions):
            if not rb.failed:self._apply_action(rb, act)

        task_counts_before = [r.tasks_done for r in self.robots]
        edge_ok = self.rng.random() >= self.p_fail
        if self.rule_variant is None:
            robot_connected = [edge_ok and self.rng.random() >= self.p_loss for _ in self.robots]
            connected = edge_ok and all(robot_connected)
        else:
            # The formal AMR model gives each robot its own communication
            # state; retain a per-robot link for R2 while keeping the existing
            # fleet-level flag for centralized reservations.
            robot_connected = [edge_ok and self.rng.random() >= self.p_loss for _ in self.robots]
            connected = all(robot_connected)
        if self.rule_variant is not None:
            for i, rb in enumerate(self.robots):
                if rb.failed:
                    continue
                robot_link = robot_connected[i]
                if rb.phase == "Pause":
                    recovered = (rb.pause_reason == "R0" and rb.authorized
                                 or rb.pause_reason == "R1" and rb.battery >= self.r1_recovery_threshold and robot_link)
                    if recovered:
                        rb.phase = "Transfer"
                        rb.pause_reason = ""
                if self.r1_enabled and rb.battery < self.r1_pause_threshold and rb.phase != "Pause":
                    rb.phase = "Pause"
                    rb.pause_reason = "R1"
                    rb.charging = True
                    rb.suspended_goal = rb.goal
                    rb.goal = min(self.chargers,
                                  key=lambda c: abs(c[0]-rb.pos[0])+abs(c[1]-rb.pos[1]))
                    self.r1_pauses_total += 1
                if not robot_link and rb.phase in ("Transfer", "Complete"):
                    if self.r2_enabled:
                        rb.phase = "Local"
                        rb.suspended_goal = rb.goal
                        rb.goal = min(self.chargers,
                                      key=lambda c: abs(c[0]-rb.pos[0])+abs(c[1]-rb.pos[1]))
                        self.r2_entries_total += 1
                    else:
                        # Preserve Transfer and let _proposal hold position.
                        pass
                elif self.r2_enabled and robot_link and rb.phase == "Local":
                    rb.phase = "Transfer"
                    if rb.suspended_goal is not None:
                        rb.goal = rb.suspended_goal
                    rb.suspended_goal = None
                if rb.phase == "Pause":
                    self.pause_steps_total += 1
                if rb.phase == "Local":
                    self.local_steps_total += 1
        proposals = [self._proposal(rb, robot_connected[i] if self.rule_variant is not None else connected)
                     for i, rb in enumerate(self.robots)]
        if self.rule_variant is not None:
            # R0 gates any proposed entry into the restricted cells and also
            # reacts to an unauthorized robot already occupying one.
            for i, (rb, path) in enumerate(zip(self.robots, proposals)):
                if not rb.authorized and (rb.pos in self.hazard or any(cell in self.hazard for cell in path[1:])):
                    proposals[i] = [rb.pos]
                    if rb.phase != "Pause" or rb.pause_reason != "R0":
                        self.r0_blocks_total += 1
                    rb.phase = "Pause"
                    rb.pause_reason = "R0"
        if self.r2_enabled:
            # Local mode has onboard obstacle sensing. Before scoring geometric
            # conflicts, locally detected interacting trajectories yield for
            # this tick; this is separate from the centralized reservation shield.
            for _ in range(self.n_agents):
                local_pairs = [pair for pair in self._conflict_pairs(proposals)
                               if any(self.robots[i].phase == "Local" for i in pair)]
                if not local_pairs:
                    break
                changed = False
                for i, j in local_pairs:
                    for k in (i, j):
                        if proposals[k] != [self.robots[k].pos]:
                            proposals[k] = [self.robots[k].pos]
                            changed = True
                    self.local_avoidances_total += 1
                if not changed:
                    break
        # Shield: serialize conflicting reservations and shorten fast moves to one cell.
        if self.shield and connected:
            reserved = {rb.pos for rb in self.robots}
            adjusted = []
            for i, path in enumerate(proposals):
                rb = self.robots[i]
                cand = path
                if len(cand) > 2:
                    cand = cand[:2]
                nxt = cand[-1]
                if nxt in reserved and nxt != rb.pos:
                    rb.interventions += 1
                    cand = [rb.pos]
                adjusted.append(cand)
                if cand[-1] != rb.pos: reserved.add(cand[-1])
            proposals = adjusted
        conflict_pairs = self._conflict_pairs(proposals)
        conflicts = {i for pair in conflict_pairs for i in pair}
        self.collision_events_total += len(conflict_pairs)
        rewards = [0.0] * self.n_agents
        for i in newly_failed:rewards[i]-=5.0
        for i, rb in enumerate(self.robots):
            if i in conflicts:
                rb.collisions += 1
                rewards[i] -= 5.0
                rb.history.append(0.0)
                # Collision state is absorbing for this time step in the simulator.
                proposals[i] = [rb.pos]
            moved = len(proposals[i]) - 1
            rb.pos = proposals[i][-1]
            if rb.charging and rb.pos in self.chargers:
                rb.battery = min(0.80, rb.battery + 0.10)
                if rb.battery >= 0.80:
                    rb.charging = False
                    rb.goal = self.sample_goal(rb.pos)
                    rb.suspended_goal = None
            else:
                cost = self.energy_per_move[rb.kind] * max(moved, 0) * (1.0 + 0.25*rb.gear)
                rb.battery = max(0.0, rb.battery - cost)
                if rb.battery <= 0.05:
                    rewards[i] -= 1.0
            # Scalar reward terms used by this simulator implementation.
            completed = rb.tasks_done - task_counts_before[i]
            rewards[i] += 4.0*completed + 2.0*moved - 0.03*(moved > 0) - 0.02*rb.gear
            link_for_risk = robot_connected[i] if self.rule_variant is not None else connected
            if self.risk_penalty and rb.gear == 2 and (not link_for_risk or self.congestion > 0.5):
                rewards[i] -= 0.25
            if not self.shield and rb.threshold < 0.30:
                rewards[i] -= 0.05

        # Sparse delivery reward is recorded on transition completion.
        # Since task completions may occur in _proposal before movement, expose totals
        # and task deltas as episode metrics rather than infer them from one scalar.
        done = self.steps >= self.horizon
        info = {
            "step": self.steps, "connected": connected,
            "collisions": self.collision_events_total,
            "collision_events": len(conflict_pairs),
            "robot_collision_involvements": sum(r.collisions for r in self.robots),
            "failed_robots": sum(int(r.failed) for r in self.robots),
            "tasks_completed": sum(r.tasks_done for r in self.robots),
            "interventions": sum(r.interventions for r in self.robots),
            "mean_battery": sum(r.battery for r in self.robots)/self.n_agents,
            "r0_blocks": self.r0_blocks_total,
            "r1_pauses": self.r1_pauses_total,
            "r2_entries": self.r2_entries_total,
            "pause_steps": self.pause_steps_total,
            "local_steps": self.local_steps_total,
            "local_avoidances": self.local_avoidances_total,
            "connected_robots": sum(int(v) for v in robot_connected),
        }
        return self.observations(), rewards, done, info
