from __future__ import annotations

import math
import queue
import threading
import time
from dataclasses import dataclass, field

from .commands import MODE_PARAMS
from .connection import Transport
from .models import Packet, Target, Vec3
from .parameters import decode_parameter, encode_parameter
from .telemetry_rates import STREAMS


@dataclass
class SimDrone:
    system_id: int
    position: Vec3
    home: Vec3
    armed: bool = False
    mode: str = "HOLD"
    landed: int = 1
    velocity: Vec3 = field(default_factory=lambda: Vec3(0, 0, 0))
    target: Target | None = None
    last_setpoint: float = 0
    stream_started: float | None = None
    dropped: bool = False
    frozen: bool = False
    reject_commands: bool = False
    params: dict = field(default_factory=lambda: {"MPC_XY_VEL_MAX": (5., 9), "COM_OF_LOSS_T": (0.5, 9), "MAV_SYS_ID": (1, 6)})


class SimTransport(Transport):
    """Deterministic kinematic test double, NOT PX4 SITL or a flight dynamics model."""
    def __init__(self, config):
        self.config = config
        self.lock = threading.RLock()
        self.incoming = queue.Queue()
        self.started = self.updated = time.monotonic()
        self.sequence = 0
        self.closed = False
        self.drones = {}
        self.stream_rates = {}
        self.stream_sent = {}
        for d in config.drones:
            p = Vec3(d.offset.x, d.offset.y, 0)
            self.drones[d.system_id] = SimDrone(d.system_id, p, p)
            self.drones[d.system_id].params["MAV_SYS_ID"] = (d.system_id, 6)

    MESSAGE_IDS = {"HEARTBEAT": 0, "SYS_STATUS": 1, "GPS_RAW_INT": 24, "ATTITUDE": 30,
                   "LOCAL_POSITION_NED": 32, "GLOBAL_POSITION_INT": 33, "VFR_HUD": 74,
                   "POSITION_TARGET_LOCAL_NED": 85, "ALTITUDE": 141, "BATTERY_STATUS": 147,
                   "ESTIMATOR_STATUS": 230, "HOME_POSITION": 242, "EXTENDED_SYS_STATE": 245}

    def _emit(self, drone, kind, _force=False, **data):
        if kind in self.MESSAGE_IDS and not _force:
            key = (drone.system_id, self.MESSAGE_IDS[kind])
            hz = self.stream_rates.get(key, STREAMS[key[1]])
            now = time.monotonic()
            previous = self.stream_sent.get(key)
            if hz <= 0 or (previous is not None and now-previous < 1/hz):
                return
            # Keep phase when scheduling is late, without emitting fake catch-up samples.
            self.stream_sent[key] = (now if previous is None else
                                     previous+max(1, int((now-previous)*hz))/hz)
        if not drone.dropped:
            self.sequence = (self.sequence+1) % 256
            self.incoming.put(Packet(drone.system_id, 1, kind, data, self.sequence))

    def receive(self, timeout):
        with self.lock:
            if self.closed:
                return None
            now = time.monotonic()
            self._update(now)
            now = time.monotonic()
            deadlines = [last+1/hz-now for key, last in self.stream_sent.items()
                         if (hz := self.stream_rates.get(key, STREAMS[key[1]])) > 0
                         and last+1/hz > now]
            wait = min(timeout, min(deadlines, default=.01))
        try:
            return self.incoming.get(timeout=max(0, wait))
        except queue.Empty:
            return None

    def _update(self, now):
        dt = min(0.1, now-self.updated)
        self.updated = now
        for drone in self.drones.values():
            if drone.mode == "OFFBOARD" and now-drone.last_setpoint > 0.5:
                drone.mode, drone.target = "HOLD", None
            if drone.armed and drone.target is not None:
                if drone.target.position is not None:
                    delta = drone.target.position-drone.position
                    speed = min(1.5, delta.norm()/max(dt, 0.001))
                    drone.velocity = delta.scale(speed/delta.norm()) if delta.norm() else Vec3(0, 0, 0)
                else:
                    drone.velocity = drone.target.velocity
                drone.position = drone.position+drone.velocity.scale(dt)
                if drone.mode == "TAKEOFF" and (drone.target.position-drone.position).norm() < 0.05:
                    drone.mode, drone.target, drone.landed = "HOLD", None, 2
                if drone.mode in ("LAND", "RTL") and drone.position.z >= -0.05:
                    drone.position = Vec3(drone.position.x, drone.position.y, 0)
                    drone.mode, drone.target, drone.landed, drone.armed = "HOLD", None, 1, False
            else:
                drone.velocity = Vec3(0, 0, 0)
        for drone in self.drones.values():
            if not drone.frozen:
                stamp = int((now-self.started)*1000)
                p, v = drone.position, drone.velocity
                self._emit(drone, "LOCAL_POSITION_NED", time_boot_ms=stamp, x=p.x, y=p.y, z=p.z, vx=v.x, vy=v.y, vz=v.z)
                self._emit(drone, "ATTITUDE", time_boot_ms=stamp, roll=0., pitch=0., yaw=0., rollspeed=0., pitchspeed=0., yawspeed=0.)
                self._emit(drone, "VFR_HUD", groundspeed=math.hypot(v.x, v.y), climb=-v.z, throttle=50 if drone.armed else 0)
                target = drone.target
                if target:
                    pos, vel = target.position, target.velocity
                    self._emit(drone, "POSITION_TARGET_LOCAL_NED", time_boot_ms=stamp, coordinate_frame=1,
                               type_mask=56 if pos else 7, x=pos.x if pos else 0., y=pos.y if pos else 0., z=pos.z if pos else 0.,
                               vx=vel.x if vel else 0., vy=vel.y if vel else 0., vz=vel.z if vel else 0.)
        for drone in self.drones.values():
            self._heartbeat(drone)
            self._emit(drone, "SYS_STATUS", onboard_control_sensors_present=0,
                       onboard_control_sensors_enabled=0, onboard_control_sensors_health=0)
            self._emit(drone, "EXTENDED_SYS_STATE", landed_state=drone.landed)
            self._emit(drone, "HOME_POSITION", altitude=10000)
            self._emit(drone, "BATTERY_STATUS", id=0, voltages=[12000]+[65535]*9, current_battery=200,
                       battery_remaining=90, current_consumed=100, temperature=2500, time_remaining=600, charge_state=1)
            self._emit(drone, "ESTIMATOR_STATUS", flags=63, vel_ratio=0.1, pos_horiz_ratio=0.1, pos_vert_ratio=0.1,
                       mag_ratio=0.1, pos_horiz_accuracy=0.05, pos_vert_accuracy=0.05)
            self._emit(drone, "GPS_RAW_INT", lat=107765000, lon=1067000000, eph=80, epv=120, fix_type=6, satellites_visible=18)
            self._emit(drone, "ALTITUDE", altitude_amsl=10-drone.position.z,
                       altitude_relative=-drone.position.z)
            self._emit(drone, "GLOBAL_POSITION_INT", time_boot_ms=int((now-self.started)*1000), lat=107765000,
                       lon=1067000000, alt=int((10-drone.position.z)*1000), relative_alt=int(-drone.position.z*1000),
                       vx=int(drone.velocity.x*100), vy=int(drone.velocity.y*100), vz=int(drone.velocity.z*100))

    def _heartbeat(self, drone, force=False):
        custom = {"POSCTL": (3, 0), "HOLD": (4, 3), "OFFBOARD": (6, 0), "TAKEOFF": (4, 2), "LAND": (4, 6), "RTL": (4, 5)}[drone.mode]
        self._emit(drone, "HEARTBEAT", _force=force, autopilot=12, type=2, base_mode=1 | (128 if drone.armed else 0),
                   custom_mode=(custom[0] << 16) | (custom[1] << 24), system_status=4 if drone.armed else 3)

    def command(self, system_id, component_id, command, params):
        with self.lock:
            drone = self.drones[system_id]
            if drone.dropped:
                return
            result = 2 if drone.reject_commands else 0
            now = time.monotonic()
            if result == 0:
                if command == 400:
                    drone.armed = bool(params[0])
                elif command == 22:
                    if not drone.armed:
                        result = 2
                    else:
                        drone.mode, drone.landed = "TAKEOFF", 3
                        drone.target = Target(system_id, Vec3(drone.position.x, drone.position.y, -(params[6]-10)))
                elif command == 176:
                    name = next((n for n, p in MODE_PARAMS.items() if params[:3] == p), None)
                    if name == "offboard":
                        if not drone.armed or drone.stream_started is None or now-drone.stream_started < 1 or now-drone.last_setpoint > 0.4:
                            result = 2
                        else:
                            drone.mode = "OFFBOARD"
                    elif name == "hold":
                        drone.mode, drone.target = "HOLD", None
                    elif name == "position":
                        drone.mode, drone.target = "POSCTL", None
                    elif name in ("land", "rtl"):
                        drone.mode, drone.landed = name.upper(), 4
                        p = drone.position if name == "land" else drone.home
                        drone.target = Target(system_id, Vec3(p.x, p.y, 0))
                    else:
                        result = 3
                elif command == 511:
                    message, interval = int(params[0]), params[1]
                    if message not in self.MESSAGE_IDS.values():
                        result = 3
                    else:
                        self.stream_rates[system_id, message] = (1e6/interval if interval > 0
                                                                 else 0 if interval == -1 else STREAMS[message])
                else:
                    result = 3
            if result != 0:
                self._emit(drone, "STATUSTEXT", severity=3, text="Simulator: command rejected", id=0, chunk_seq=0)
            self._emit(drone, "COMMAND_ACK", command=command, result=result, progress=100)
            if command != 511:
                self._heartbeat(drone, force=True)

    def setpoint(self, target, component_id):
        with self.lock:
            drone, now = self.drones[target.system_id], time.monotonic()
            if drone.dropped:
                return
            if now-drone.last_setpoint > 0.4:
                drone.stream_started = now
            drone.last_setpoint = now
            if drone.mode == "OFFBOARD":
                drone.target = target

    def parameter(self, system_id, component_id, operation, name="", value=0, param_type=9):
        with self.lock:
            drone = self.drones[system_id]
            if operation == "set" and name in drone.params:
                drone.params[name] = (decode_parameter(value, param_type), param_type)
            for index, (key, (v, t)) in enumerate(drone.params.items()):
                if operation == "list" or key == name:
                    self._emit(drone, "PARAM_VALUE", param_id=key, param_value=encode_parameter(v, t), param_type=t,
                               param_index=index, param_count=len(drone.params))

    def control(self, operation, system_id, values):
        with self.lock:
            drone = self.drones[system_id]
            if operation == "move":
                if not drone.armed or drone.landed != 2:
                    raise ValueError("Sim drone must first be armed and airborne")
                if drone.mode == "OFFBOARD":
                    raise ValueError("Use formation stop before manually moving a follower")
                drone.target = Target(system_id, Vec3.parse(values))
            elif operation in ("drop", "freeze", "reject"):
                if values not in (["on"], ["off"]):
                    raise ValueError("Expected on or off")
                setattr(drone, {"drop": "dropped", "freeze": "frozen", "reject": "reject_commands"}[operation], values == ["on"])
            else:
                raise ValueError("sim move|drop|freeze|reject ID ...")

    def close(self):
        self.closed = True
