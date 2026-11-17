import math
import tempfile
import time
import unittest
from collections import deque
from types import SimpleNamespace

from support import FakeTransport, MemoryLog, Packet, Registry, Telemetry, heartbeat, local, wait_until
from multi_drone.app import App
from multi_drone.commands import CommandService
from multi_drone.config import Config
from multi_drone.plot_data import PlotHistory, snapshot_frame
from multi_drone.telemetry_rates import DEFAULT_HZ, MAX_HZ, TelemetryRates


class RateRequestTests(unittest.TestCase):
    def test_group_limits_and_target_isolation(self):
        rates = TelemetryRates()
        rates.discover(3)
        rates.discover(4)
        for group, maximum in MAX_HZ.items():
            for hz in range(5, maximum+1, 5):
                rates.set_group(3, group, hz)
            for hz in (0, 4, 7, maximum+5, True, 10., float("nan")):
                with self.assertRaises(ValueError):
                    rates.set_group(3, group, hz)
        self.assertEqual(rates.requested[4], DEFAULT_HZ)
        with self.assertRaises(ValueError):
            rates.set_group(99, "vehicle", 20)

    def test_serial_ack_rejection_timeout_and_coalesced_edits(self):
        rates, transport = TelemetryRates(), FakeTransport()
        commands = CommandService(transport, MemoryLog())
        rates.discover(3)
        rates.pending.clear()
        rates.set_group(3, "vehicle", 100)
        states = {3: SimpleNamespace(component_id=1)}
        rates.tick(commands, states)
        first = transport.sent[-1][1]
        self.assertEqual(first[:3], (3, 1, 511))
        self.assertEqual(first[3], [30., 10000.])
        rates.tick(commands, states)
        self.assertEqual(len(transport.sent), 1)
        rates.set_group(3, "vehicle", 50)
        rates.set_group(3, "vehicle", 75)
        commands.handle(Packet(3, 1, "COMMAND_ACK", {"command": 511, "result": 3}))
        rates.tick(commands, states)
        self.assertEqual(transport.sent[-1][1][3], [33., 1e6/75])
        # All queued messages, including the previously active one, use the newest edit.
        self.assertTrue(all(hz == 75 for hz in rates.pending.values()))
        commands.tick(time.monotonic()+5)
        rates.tick(commands, states)
        info = rates.snapshot()[3]["vehicle"]
        self.assertIn("No final ACK", info["messages"][33])

    def test_simulator_applies_rates_per_drone_and_rejects_old_connection(self):
        with tempfile.TemporaryDirectory() as directory:
            app = App(Config.default(), directory)
            try:
                app.connect("sim")
                app.wait_for_vehicles(3)
                app.set_telemetry_rate(1, "localPosition", 50, id(app.registry))
                wait_until(lambda: app.transport.stream_rates.get((1, 32)) == 50)
                self.assertEqual(app.telemetry_rates.requested[2]["localPosition"], 20)
                with self.assertRaises(ValueError):
                    app.set_telemetry_rate(1, "localPosition", 25, -1)
            finally:
                app.close()


class SampleBatchTests(unittest.TestCase):
    def setUp(self):
        self.registry = Registry()
        self.telemetry = Telemetry(self.registry)
        self.app = SimpleNamespace(registry=self.registry, telemetry=self.telemetry,
                                   transport=object(), endpoint="test", config=Config.default(),
                                   telemetry_rates=TelemetryRates())
        self.telemetry.ingest(heartbeat(3), 10.)
        self.telemetry.ingest(heartbeat(4), 10.)

    def test_100hz_and_50hz_survive_10hz_display_batches_without_duplicates(self):
        history, cursor = PlotHistory(), (None, 0)
        for batch in range(10):
            for i in range(batch*10, (batch+1)*10):
                now = 10.01+i*.01
                self.telemetry.ingest(Packet(3, 1, "ATTITUDE", dict(
                    time_boot_ms=10+i*10, roll=i*.001, pitch=0., yaw=0.,
                    rollspeed=0., pitchspeed=0., yawspeed=0.)), now)
                if i % 2 == 0:
                    self.telemetry.ingest(local(4, stamp=10+i*10, x=i), now)
            data = snapshot_frame(self.app, now, cursor)
            history.update(data)
            history.update(data)  # Duplicate delivery cannot duplicate samples.
            cursor = data["sample_cursor"]
        self.assertEqual(len(history.series(3, "vehicle.roll")[0]), 100)
        self.assertEqual(len(history.series(4, "localPosition.x")[0]), 50)
        self.assertAlmostEqual(history.measured_rates(3, ["vehicle.roll"], now)["vehicle.roll"], 100)
        self.assertAlmostEqual(history.measured_rates(4, ["localPosition.x"], now)["localPosition.x"], 50)
        history.update(snapshot_frame(self.app, now+.1, cursor))
        self.assertEqual(len(history.series(3, "vehicle.roll")[0]), 100)

    def test_backpressure_retry_and_bounded_overflow_are_explicit(self):
        self.telemetry.sample_batches = deque(maxlen=3)
        cursor = (id(self.registry), self.telemetry.sample_sequence)
        for i in range(6):
            self.telemetry.ingest(local(3, stamp=i*10, x=i), 10+i*.01)
        first = snapshot_frame(self.app, 10.1, cursor)
        retry = snapshot_frame(self.app, 10.2, cursor)
        self.assertEqual(first["samples"], retry["samples"])
        self.assertEqual(first["missed_batches"], 3)
        history = PlotHistory()
        history.update(retry)
        self.assertEqual(history.series(3, "localPosition.x")[1], [3, 4, 5])
        self.assertEqual(history.missed_batches, 3)
        history.update(retry)
        self.assertEqual(history.missed_batches, 3)

    def test_full_120_second_history_at_100hz_and_epoch_reset(self):
        history = PlotHistory()
        data = snapshot_frame(self.app, 10., (None, 0))
        data["samples"] = [(i+1, 10+i*.01, 3, [("vehicle.roll", i)]) for i in range(12000)]
        data["now"] = 129.99
        history.update(data)
        x, y = history.series(3, "vehicle.roll")
        self.assertEqual(sum(math.isfinite(v) for v in y), 12000)
        data.update(epoch=-1, samples=[(1, 130., 3, [("vehicle.roll", 9)])], now=130.)
        history.update(data)
        self.assertEqual(history.series(3, "vehicle.roll")[1][0], 9)
        self.assertEqual(history.sample_sequence, 1)

    def test_altitude_sources_do_not_double_count(self):
        self.telemetry.ingest(Packet(3, 1, "GLOBAL_POSITION_INT", dict(
            time_boot_ms=10, lat=0, lon=0, alt=1000, relative_alt=500, vx=0, vy=0, vz=0)), 10.1)
        before = self.telemetry.sample_sequence
        self.telemetry.ingest(Packet(3, 1, "ALTITUDE", dict(altitude_amsl=2, altitude_relative=1)), 10.11)
        self.assertEqual(self.telemetry.sample_sequence, before)
        self.telemetry.ingest(Packet(3, 1, "ALTITUDE", dict(altitude_amsl=2, altitude_relative=1)), 11.2)
        self.assertEqual(self.telemetry.sample_sequence, before+1)

    def test_quaternion_fallback_does_not_double_sample_euler_stream(self):
        self.telemetry.ingest(Packet(3, 1, "ATTITUDE", dict(time_boot_ms=10, roll=0., pitch=0., yaw=0.,
                              rollspeed=0., pitchspeed=0., yawspeed=0.)), 10.1)
        before = self.telemetry.sample_sequence
        data = dict(time_boot_ms=10, q1=1., q2=0., q3=0., q4=0., rollspeed=0., pitchspeed=0., yawspeed=0.)
        self.telemetry.ingest(Packet(3, 1, "ATTITUDE_QUATERNION", data), 10.11)
        self.assertEqual(self.telemetry.sample_sequence, before)
        self.telemetry.ingest(Packet(3, 1, "ATTITUDE_QUATERNION", {**data, "time_boot_ms": 20}), 10.7)
        self.assertEqual(self.telemetry.sample_sequence, before+1)
