#!/usr/bin/env python3
"""
CARLA Multi-LiDAR Roadside Data Collector
CARLA roadside multi-LiDAR / camera data collection

Spawns 4 roadside LiDARs (+ optional co-located RGB cameras for REV8 simulation)
at an intersection in synchronous mode. Collects synchronized frames per weather
condition.

Point clouds: float32 numpy arrays (N, 4) — [x, y, z, intensity] in sensor-local
frame. World poses saved per-frame in metadata.json for downstream transformation.

Output layout:
    carla_data/
    ├── clear/
    │   ├── frame_000000/
    │   │   ├── lidar_ne.npy   lidar_nw.npy   lidar_se.npy   lidar_sw.npy
    │   │   ├── camera_ne.png  ...              (only with --cameras)
    │   │   ├── gt_boxes.json
    │   │   └── metadata.json
    │   └── ...
    ├── rain_med/  fog/  night/
    └── background/   (only with --background)

Usage:
    python collect_data.py --verify              # 10-frame sanity check
    python collect_data.py --condition clear     # single condition, 2000 frames
    python collect_data.py --cameras             # + co-located RGB cameras (REV8)
    python collect_data.py --background          # no traffic (FRGB3D warmup data)
    python collect_data.py                       # all conditions, 2000 frames each
"""

import argparse
import json
import math
import queue
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

try:
    import carla
except ImportError:
    sys.exit("ERROR: carla module not found. Install the CARLA Python API egg/wheel.")

# Image saving: prefer cv2 → PIL → npy fallback
try:
    import cv2 as _cv2
    HAS_CV2 = True
    HAS_PIL = False
except ImportError:
    HAS_CV2 = False
    try:
        from PIL import Image as _PIL
        HAS_PIL = True
    except ImportError:
        HAS_PIL = False

if not HAS_CV2 and not HAS_PIL:
    print("WARNING: neither opencv-python nor Pillow found — cameras saved as .npy")


def _save_png(path: Path, bgra: np.ndarray):
    """Save BGRA numpy array as PNG (or .npy if no image library available)."""
    if HAS_CV2:
        _cv2.imwrite(str(path), bgra[:, :, :3])          # cv2 writes BGR
    elif HAS_PIL:
        _PIL.fromarray(bgra[:, :, [2, 1, 0]]).save(str(path))   # BGR→RGB
    else:
        np.save(str(path).replace(".png", ".npy"), bgra[:, :, :3])


# ── Sensor geometry ────────────────────────────────────────────────────────────

_CORNERS: Dict[str, tuple] = {
    "ne": (+20.0, +20.0),
    "nw": (-20.0, +20.0),
    "se": (+20.0, -20.0),
    "sw": (-20.0, -20.0),
}
_MOUNT_HEIGHT = 5.0    # m above ground plane (typical traffic-light pole)
_MOUNT_PITCH  = -15.0  # degrees, tilt toward intersection center


def sensor_transform(center: carla.Location, dx: float, dy: float) -> carla.Transform:
    """Place a sensor at (center + (dx, dy, HEIGHT)), yaw pointing toward center."""
    yaw = math.degrees(math.atan2(-dy, -dx))   # vector from corner toward center
    return carla.Transform(
        carla.Location(x=center.x + dx, y=center.y + dy, z=center.z + _MOUNT_HEIGHT),
        carla.Rotation(pitch=_MOUNT_PITCH, yaw=yaw, roll=0.0),
    )


def transform_to_dict(t: carla.Transform) -> dict:
    l, r = t.location, t.rotation
    return {"x": l.x, "y": l.y, "z": l.z,
            "pitch": r.pitch, "yaw": r.yaw, "roll": r.roll}


# ── LiDAR attributes — OS-1 matched ───────────────────────────────────────────

_LIDAR_ATTRS = {
    "channels":                    "128",
    "range":                       "100.0",
    "points_per_second":           "1310720",
    "rotation_frequency":          "10.0",
    "upper_fov":                   "22.5",
    "lower_fov":                   "-22.5",
    "horizontal_fov":              "360.0",
    "atmosphere_attenuation_rate": "0.004",
    "dropoff_general_rate":        "0.45",
    "dropoff_intensity_limit":     "0.8",
    "dropoff_zero_intensity":      "0.4",
    "noise_stddev":                "0.01",
}

_CAMERA_ATTRS = {
    "image_size_x": "1280",
    "image_size_y": "720",
    "fov":          "90",
}


# ── Weather presets ────────────────────────────────────────────────────────────

ALL_CONDITIONS = ["clear", "rain_med", "fog", "night"]

_WEATHER_PARAMS = {
    "clear": dict(cloudiness=10, precipitation=0, precipitation_deposits=0,
                  wind_intensity=10, sun_azimuth_angle=70, sun_altitude_angle=70,
                  fog_density=0, fog_distance=0, wetness=0),
    "rain_med": dict(cloudiness=80, precipitation=50, precipitation_deposits=40,
                     wind_intensity=40, sun_azimuth_angle=70, sun_altitude_angle=30,
                     fog_density=10, fog_distance=0.75, wetness=80),
    "fog": dict(cloudiness=60, precipitation=0, precipitation_deposits=0,
                wind_intensity=5, sun_azimuth_angle=70, sun_altitude_angle=30,
                fog_density=80, fog_distance=30, wetness=10),
    "night": dict(cloudiness=30, precipitation=0, precipitation_deposits=0,
                  wind_intensity=10, sun_azimuth_angle=180, sun_altitude_angle=-30,
                  fog_density=5, fog_distance=0, wetness=20),
}


# ── Intersection finder ────────────────────────────────────────────────────────

def find_intersection(world: carla.World) -> carla.Location:
    """Return the location of the most-connected junction (busiest intersection)."""
    spawn_points = world.get_map().get_spawn_points()
    waypoints = world.get_map().generate_waypoints(5.0)

    best_loc, best_score = carla.Location(x=0, y=0, z=0), 0
    visited = set()

    for wp in waypoints:
        if not getattr(wp, "is_junction", False):
            continue
        loc = wp.transform.location
        key = (round(loc.x, 1), round(loc.y, 1))
        if key in visited:
            continue
        visited.add(key)

        score = sum(1 for sp in spawn_points if sp.location.distance(loc) < 60.0)
        if score > best_score:
            best_score, best_loc = score, loc

    print(f"Auto-selected intersection: ({best_loc.x:.1f}, {best_loc.y:.1f}, {best_loc.z:.1f})"
          f"  [{best_score} spawn points within 60 m]")
    return best_loc


# ── SensorManager ──────────────────────────────────────────────────────────────

_CAM_360_OFFSETS = [0, 90, 180, 270]   # yaw offsets for full 360° coverage


class SensorManager:
    """Lifecycle management for roadside LiDARs and optional co-located cameras.

    cameras=True     : 1 camera per corner facing intersection (original mode)
    cameras_360=True : 4 cameras per corner at 0°/90°/180°/270° yaw offsets
                       giving full 360° color coverage (REV8-like)
    """

    def __init__(self, world: carla.World, center: carla.Location,
                 cameras: bool = False, cameras_360: bool = False):
        self.world       = world
        self.center      = center
        self.cameras     = cameras
        self.cameras_360 = cameras_360

        self.actors:     Dict[str, carla.Actor]     = {}
        self.queues:     Dict[str, queue.Queue]     = {}
        self.transforms: Dict[str, carla.Transform] = {}  # keyed by corner

    def setup(self):
        bp_lib = self.world.get_blueprint_library()

        lidar_bp = bp_lib.find("sensor.lidar.ray_cast")
        for k, v in _LIDAR_ATTRS.items():
            lidar_bp.set_attribute(k, v)

        cam_bp = None
        if self.cameras or self.cameras_360:
            cam_bp = bp_lib.find("sensor.camera.rgb")
            for k, v in _CAMERA_ATTRS.items():
                cam_bp.set_attribute(k, v)

        for corner, (dx, dy) in _CORNERS.items():
            t = sensor_transform(self.center, dx, dy)
            self.transforms[corner] = t

            lidar = self.world.spawn_actor(lidar_bp, t)
            q_l = queue.Queue()
            lidar.listen(q_l.put)
            self.actors[f"lidar_{corner}"] = lidar
            self.queues[f"lidar_{corner}"] = q_l

            if cam_bp is None:
                continue

            if self.cameras_360:
                # 4 cameras at 90° intervals — full 360° coverage
                for yaw_off in _CAM_360_OFFSETS:
                    cam_t = carla.Transform(
                        t.location,
                        carla.Rotation(pitch=t.rotation.pitch,
                                       yaw=t.rotation.yaw + yaw_off,
                                       roll=t.rotation.roll),
                    )
                    name = f"camera_{corner}_{yaw_off}"
                    cam = self.world.spawn_actor(cam_bp, cam_t)
                    q_c = queue.Queue()
                    cam.listen(q_c.put)
                    self.actors[name] = cam
                    self.queues[name] = q_c
            else:
                # Single camera facing intersection center
                cam = self.world.spawn_actor(cam_bp, t)
                q_c = queue.Queue()
                cam.listen(q_c.put)
                self.actors[f"camera_{corner}"] = cam
                self.queues[f"camera_{corner}"] = q_c

        if self.cameras_360:
            kind = "LiDAR + 4×cameras (360°)"
        elif self.cameras:
            kind = "LiDAR + camera"
        else:
            kind = "LiDAR only"
        print(f"Spawned {len(self.actors)} sensors ({kind}).")

    def get_frame(self, timeout: float = 5.0) -> Dict:
        """Collect one synchronized frame from all sensors."""
        data = {}
        for name, q in self.queues.items():
            try:
                data[name] = q.get(timeout=timeout)
            except queue.Empty:
                raise RuntimeError(f"Sensor '{name}' timed out after {timeout:.1f}s — "
                                   "CARLA may be overloaded or a sensor failed.")
        return data

    def destroy(self):
        for actor in self.actors.values():
            if actor.is_alive:
                actor.stop()
                actor.destroy()
        self.actors.clear()
        self.queues.clear()
        print("Sensors destroyed.")


# ── Traffic spawner ────────────────────────────────────────────────────────────

class TrafficSpawner:
    def __init__(self, client: carla.Client, world: carla.World):
        self.client      = client
        self.world       = world
        self.actors:      List[carla.Actor] = []   # vehicles
        self.walkers:     List[carla.Actor] = []   # pedestrian pawns
        self.controllers: List[carla.Actor] = []   # AI walker controllers

    def spawn(self, center: carla.Location,
              n_vehicles: int = 30, n_walkers: int = 0,
              radius: float = 80.0):
        bp_lib = self.world.get_blueprint_library()

        # Vehicles
        veh_bps = bp_lib.filter("vehicle.*")
        all_sps = self.world.get_map().get_spawn_points()
        near_sps = [sp for sp in all_sps if sp.location.distance(center) < radius]
        sps = near_sps if near_sps else all_sps

        count = 0
        for i, sp in enumerate(sps[: n_vehicles + 10]):
            if count >= n_vehicles:
                break
            v = self.world.try_spawn_actor(veh_bps[i % len(veh_bps)], sp)
            if v is not None:
                v.set_autopilot(True)
                self.actors.append(v)
                count += 1
        print(f"Spawned {count}/{n_vehicles} vehicles (autopilot ON).")

        # Walkers
        if n_walkers > 0:
            self._spawn_walkers(bp_lib, n_walkers)

    def _spawn_walkers(self, bp_lib, n_walkers: int):
        SpawnActor = carla.command.SpawnActor

        walker_bps = bp_lib.filter("walker.pedestrian.*")
        ctrl_bp    = bp_lib.find("controller.ai.walker")

        # Batch-spawn pedestrian pawns
        batch, spawn_locs = [], []
        for _ in range(n_walkers):
            loc = self.world.get_random_location_from_navigation()
            if loc is None:
                continue
            bp = random.choice(walker_bps)
            if bp.has_attribute("is_invincible"):
                bp.set_attribute("is_invincible", "false")
            batch.append(SpawnActor(bp, carla.Transform(loc)))
            spawn_locs.append(loc)

        results   = self.client.apply_batch_sync(batch, True)
        walker_ids = [r.actor_id for r in results if not r.error]
        self.walkers = [self.world.get_actor(wid) for wid in walker_ids
                        if self.world.get_actor(wid) is not None]

        # Attach AI controllers (need one tick before start())
        batch = [SpawnActor(ctrl_bp, carla.Transform(), wid)
                 for wid in walker_ids]
        results = self.client.apply_batch_sync(batch, True)
        ctrl_ids = [r.actor_id for r in results if not r.error]

        self.world.tick()   # controllers must exist in the world before start()

        self.controllers = []
        for cid in ctrl_ids:
            ctrl = self.world.get_actor(cid)
            if ctrl is None:
                continue
            ctrl.start()
            ctrl.go_to_location(self.world.get_random_location_from_navigation())
            ctrl.set_max_speed(1.4)   # ~5 km/h
            self.controllers.append(ctrl)

        print(f"Spawned {len(self.walkers)}/{n_walkers} walkers (AI ON).")

    def destroy(self):
        # Stop controllers first, then destroy walkers + controllers together
        for ctrl in self.controllers:
            if ctrl.is_alive:
                ctrl.stop()
        self.client.apply_batch(
            [carla.command.DestroyActor(a.id)
             for a in self.controllers + self.walkers + self.actors]
        )
        self.actors.clear()
        self.walkers.clear()
        self.controllers.clear()


# ── Ground truth ───────────────────────────────────────────────────────────────

def collect_gt(world: carla.World, center: carla.Location, radius: float = 80.0) -> List[dict]:
    """Return GT bounding boxes for all vehicles and pedestrians within radius."""
    objects = []
    filters = ["vehicle.*", "walker.pedestrian.*"]
    for f in filters:
        for actor in world.get_actors().filter(f):
            loc = actor.get_location()
            if loc.distance(center) > radius:
                continue
            t  = actor.get_transform()
            bb = actor.bounding_box
            objects.append({
                "id":    actor.id,
                "class": actor.type_id,
                "bbox": {
                    "center":     [t.location.x, t.location.y, t.location.z],
                    "dimensions": [bb.extent.x * 2, bb.extent.y * 2, bb.extent.z * 2],
                    "rotation":   [t.rotation.pitch, t.rotation.yaw, t.rotation.roll],
                },
                "distance_from_center": loc.distance(center),
            })
    return objects


# ── Data writer ────────────────────────────────────────────────────────────────

class DataWriter:
    def __init__(self, root: Path, condition: str):
        self.root = root / condition
        self.root.mkdir(parents=True, exist_ok=True)

    def write(self, idx: int, frame_data: Dict,
              transforms: Dict, world: carla.World,
              center: carla.Location, condition: str):
        d = self.root / f"frame_{idx:06d}"
        d.mkdir(exist_ok=True)

        first = next(iter(frame_data.values()))

        for name, meas in frame_data.items():
            if name.startswith("lidar_"):
                pts = np.frombuffer(meas.raw_data, dtype=np.float32).reshape(-1, 4)
                np.save(d / f"{name}.npy", pts)
            elif name.startswith("camera_"):
                img = np.frombuffer(meas.raw_data, dtype=np.uint8)
                img = img.reshape((meas.height, meas.width, 4))   # BGRA
                _save_png(d / f"{name}.png", img)

        gt = collect_gt(world, center)
        (d / "gt_boxes.json").write_text(json.dumps({
            "frame_id":  f"frame_{idx:06d}",
            "condition": condition,
            "objects":   gt,
        }, indent=2))

        (d / "metadata.json").write_text(json.dumps({
            "frame_id":    f"frame_{idx:06d}",
            "carla_frame": first.frame,
            "timestamp":   first.timestamp,
            "condition":   condition,
            "sensor_poses": {corner: transform_to_dict(t)
                             for corner, t in transforms.items()},
        }, indent=2))


# ── Main collector ─────────────────────────────────────────────────────────────

class Collector:
    def __init__(self, args):
        self.args  = args
        self.client: Optional[carla.Client] = None
        self.world:  Optional[carla.World]  = None
        self._original_settings = None

    # ── connection & world ──

    def connect(self):
        print(f"Connecting to CARLA at {self.args.host}:{self.args.port} …")
        self.client = carla.Client(self.args.host, self.args.port)
        self.client.set_timeout(30.0)
        self.world = self.client.get_world()
        print(f"Connected.  Server: {self.client.get_server_version()}"
              f"  Map: {self.world.get_map().name}")

    def load_town(self):
        if self.args.town not in self.world.get_map().name:
            print(f"Loading {self.args.town} …")
            self.world = self.client.load_world(self.args.town)
            time.sleep(5.0)
            print(f"Loaded {self.args.town}.")

    def enable_sync(self):
        settings = self.world.get_settings()
        self._original_settings = settings
        settings.synchronous_mode    = True
        settings.fixed_delta_seconds = 0.1   # 10 Hz
        self.world.apply_settings(settings)
        self.client.get_trafficmanager().set_synchronous_mode(True)
        print("Synchronous mode ON (10 Hz).")

    def disable_sync(self):
        if self._original_settings is not None:
            self.world.apply_settings(self._original_settings)
            print("World settings restored.")

    def get_center(self) -> carla.Location:
        if self.args.center:
            x, y, z = self.args.center
            return carla.Location(x=x, y=y, z=z)
        return find_intersection(self.world)

    # ── per-condition collection ──

    def run_condition(self, condition: str, n_frames: int,
                      out_root: Path, no_traffic: bool = False):
        sep = "─" * 60
        print(f"\n{sep}")
        print(f"  Condition : {condition}")
        print(f"  Frames    : {n_frames}")
        print(f"  Traffic   : {'none (background mode)' if no_traffic else f'{self.args.num_vehicles} vehicles  {self.args.num_walkers} walkers'}")
        if self.args.cameras_360:
            cam_label = "yes — 4×cameras 360° (full REV8 mode)"
        elif self.args.cameras:
            cam_label = "yes — 1×camera (partial REV8 mode)"
        else:
            cam_label = "no"
        print(f"  Cameras   : {cam_label}")
        print(f"{sep}")

        center = self.get_center()
        mgr = SensorManager(self.world, center,
                            cameras=self.args.cameras,
                            cameras_360=self.args.cameras_360)
        mgr.setup()

        traffic: Optional[TrafficSpawner] = None
        if not no_traffic:
            traffic = TrafficSpawner(self.client, self.world)
            traffic.spawn(center, n_vehicles=self.args.num_vehicles,
                          n_walkers=self.args.num_walkers)

        self.world.set_weather(carla.WeatherParameters(**_WEATHER_PARAMS[condition]))
        print(f"Weather: {condition}")

        # Warmup — let NPC vehicles settle, drain sensor queues
        print(f"Warmup: {self.args.warmup} ticks …")
        for _ in range(self.args.warmup):
            self.world.tick()
            mgr.get_frame()

        writer = DataWriter(out_root, condition)
        t0 = time.time()
        print(f"Collecting {n_frames} frames …")

        for i in range(n_frames):
            self.world.tick()
            frame_data = mgr.get_frame()
            writer.write(i, frame_data, mgr.transforms, self.world, center, condition)

            if i == 0 or (i + 1) % 100 == 0:
                elapsed = time.time() - t0
                fps = (i + 1) / elapsed
                eta = (n_frames - i - 1) / fps if fps > 0 else 0
                print(f"  [{i+1:>5}/{n_frames}]  {fps:.1f} fps  ETA {eta:.0f}s")

        mgr.destroy()
        if traffic:
            traffic.destroy()
        print(f"Saved → {out_root / condition}")

    # ── top-level run ──

    def run(self):
        try:
            self.connect()
            self.load_town()
            self.enable_sync()

            n_frames  = 10 if self.args.verify else self.args.frames
            out_root  = Path(self.args.output_dir)
            if self.args.background:
                out_root = out_root / "background"

            if self.args.condition:
                conditions = [self.args.condition]
            else:
                conditions = ALL_CONDITIONS

            for cond in conditions:
                self.run_condition(cond, n_frames, out_root,
                                   no_traffic=self.args.background)

            print("\nAll conditions complete.")

        except KeyboardInterrupt:
            print("\nInterrupted by user.")
        except Exception:
            import traceback
            traceback.print_exc()
        finally:
            self.disable_sync()


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="CARLA multi-LiDAR roadside data collector"
    )
    p.add_argument("--host",         default="localhost", help="CARLA server host")
    p.add_argument("--port",         type=int, default=2000)
    p.add_argument("--town",         default="Town10HD",
                   help="CARLA town to load (default: Town10HD)")
    p.add_argument("--output-dir",   default="carla_data",
                   help="Root output directory (default: carla_data/)")
    p.add_argument("--condition",    choices=ALL_CONDITIONS,
                   help="Collect a single weather condition (default: all)")
    p.add_argument("--frames",       type=int, default=2000,
                   help="Frames per condition (default: 2000)")
    p.add_argument("--warmup",       type=int, default=50,
                   help="Warmup ticks before recording (default: 50)")
    p.add_argument("--num-vehicles", type=int, default=30,
                   help="NPC vehicles to spawn (default: 30)")
    p.add_argument("--num-walkers",  type=int, default=20,
                   help="NPC pedestrians to spawn (default: 20)")
    p.add_argument("--cameras",      action="store_true",
                   help="1 co-located RGB camera per corner (partial REV8 mode)")
    p.add_argument("--cameras-360",  action="store_true",
                   help="4 cameras per corner at 0/90/180/270° — full 360° color")
    p.add_argument("--background",   action="store_true",
                   help="No traffic — collect background frames for FRGB3D warmup")
    p.add_argument("--verify",       action="store_true",
                   help="Quick 10-frame sanity check (overrides --frames)")
    p.add_argument("--center",       nargs=3, type=float, metavar=("X", "Y", "Z"),
                   help="Intersection center world coordinates (auto-detect if omitted)")
    return p.parse_args()


if __name__ == "__main__":
    Collector(parse_args()).run()
