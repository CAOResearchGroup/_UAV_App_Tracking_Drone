"""Per-vehicle stream requests, serialized because COMMAND_ACK only identifies command 511."""
from collections import OrderedDict
import threading


GROUP_STREAMS = {
    "vehicle": (30, 33, 74, 141),
    "gps": (24,), "localPosition": (32,), "localPositionSetpoint": (85,),
    "estimatorStatus": (230,), "battery": (147,),
}
MAX_HZ = {name: 100 if name == "vehicle" else 50 if name.startswith("localPosition") else 30
          for name in GROUP_STREAMS}
DEFAULT_HZ = {name: 20 if name in ("vehicle", "localPosition") else 5 for name in GROUP_STREAMS}
STREAMS = {0: 1, 1: 2, 242: .5, 245: 2,
           **{message: DEFAULT_HZ[group] for group, messages in GROUP_STREAMS.items() for message in messages}}


class TelemetryRates:
    def __init__(self):
        self.lock = threading.RLock()
        self.pending = OrderedDict()
        self.requested = {}
        self.results = {}
        self.active = None

    def discover(self, system_id):
        with self.lock:
            for message, hz in STREAMS.items():
                self.pending[system_id, message] = hz
            self.requested[system_id] = dict(DEFAULT_HZ)

    def set_group(self, system_id, group, hz):
        if group not in GROUP_STREAMS:
            raise ValueError("Unknown telemetry group")
        if isinstance(hz, bool) or not isinstance(hz, int) or hz < 5 or hz > MAX_HZ[group] or hz % 5:
            raise ValueError(f"{group}: choose 5..{MAX_HZ[group]} Hz in steps of 5")
        with self.lock:
            if system_id not in self.requested:
                raise ValueError("Drone has not been discovered")
            self.requested[system_id][group] = hz
            for message in GROUP_STREAMS[group]:
                self.pending[system_id, message] = hz
                self.results.pop((system_id, message), None)

    def tick(self, commands, states):
        with self.lock:
            if self.active:
                key, hz, future = self.active
                if not future.done():
                    return
                try:
                    result = future.result()
                    status = "Đã chấp nhận" if result["acknowledged"] else result.get("result_name", "Từ chối")
                except Exception as exc:
                    status = str(exc)
                self.results[key] = (hz, status)
                self.active = None
            if self.pending:
                key, hz = self.pending.popitem(last=False)
                system_id, message = key
                state = states.get(system_id)
                if state:
                    try:
                        future = commands.send(system_id, state.component_id, 511,
                                               [float(message), 1e6/hz], "request-stream", timeout=1)
                        self.active = (key, hz, future)
                    except Exception as exc:
                        self.results[key] = (hz, str(exc))

    def snapshot(self):
        with self.lock:
            return {n: {group: {"hz": hz, "messages": {
                message: ("Đang yêu cầu" if (n, message) in self.pending or
                          (self.active and self.active[0] == (n, message)) else
                          self.results.get((n, message), (hz, "Chưa yêu cầu"))[1])
                for message in GROUP_STREAMS[group]}}
                for group, hz in groups.items()} for n, groups in self.requested.items()}
