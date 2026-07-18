"""
pump_controller.py -- Closed-loop pump control over a serial/USB link.

The Raspberry Pi reads the liquid volume every frame (see liquid_level.py) and
this module turns that stream of readings into pump commands so the pump stops
at a precise target volume.

Key features
------------
* Continuous streaming: every measurement is sent to the pump/host so it
  "continuously receives input" as requested.
* Predictive stop: the pump keeps running until
      volume >= target - flow_rate * stop_latency - safety_margin
  i.e. it compensates for the liquid still in transit and the pump's own stop
  latency, so it does not overshoot the target.
* Hysteresis / latch: once the target is reached the pump stays stopped and
  will not chatter around the setpoint.
* Hardware-agnostic link: SerialPumpLink (pyserial) for the real pump,
  MockPumpLink for tests / dry runs. Command strings are configurable so you
  can match whatever protocol your pump firmware speaks.
* Backward-compatible query mode: still answers the old "print" / "autoPrint" /
  "stopPrint" text commands from the previous firmware if you use them.

IMPORTANT: pumps differ. Set PumpConfig.cmd_run / cmd_stop (and optionally
cmd_setpoint / stream_format) to the exact bytes your pump expects. The defaults
send human-readable ASCII lines ("RUN\\n", "STOP\\n", "V:12.34\\n").
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class PumpState(Enum):
    IDLE = "idle"        # no target set / not dosing
    RUNNING = "running"  # actively pumping toward target
    REACHED = "reached"  # target hit, latched off
    PAUSED = "paused"    # manually paused


class Direction(Enum):
    FILL = "fill"        # volume increases toward target
    DRAIN = "drain"      # volume decreases toward target


# ---------------------------------------------------------------------------
# Serial links
# ---------------------------------------------------------------------------

class PumpLink:
    """Abstract byte link to the pump."""

    def write_line(self, text: str) -> None:
        raise NotImplementedError

    def read_available(self) -> str:
        return ""

    def close(self) -> None:
        pass


class MockPumpLink(PumpLink):
    """In-memory link for tests and dry runs; records everything sent."""

    def __init__(self):
        self.sent = []
        self._inbox = ""

    def write_line(self, text: str) -> None:
        self.sent.append(text)

    def feed(self, text: str) -> None:
        """Simulate bytes arriving from the pump/host."""
        self._inbox += text

    def read_available(self) -> str:
        data, self._inbox = self._inbox, ""
        return data

    @property
    def last(self) -> Optional[str]:
        return self.sent[-1] if self.sent else None


class SerialPumpLink(PumpLink):
    """pyserial-backed link. Import of serial is lazy so tests need no hardware."""

    def __init__(self, port: str, baud: int = 9600, timeout: float = 0.0):
        import serial  # lazy
        self.ser = serial.Serial(port, baud, timeout=timeout)

    def write_line(self, text: str) -> None:
        self.ser.write(text.encode("ascii", errors="ignore"))

    def read_available(self) -> str:
        try:
            n = self.ser.in_waiting
        except Exception:
            n = 0
        if n:
            try:
                return self.ser.read(n).decode("ascii", errors="ignore")
            except Exception:
                return ""
        return ""

    def close(self) -> None:
        try:
            self.ser.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------

@dataclass
class PumpConfig:
    direction: Direction = Direction.FILL
    target_ml: float = 0.0
    tolerance_ml: float = 0.1          # counted "reached" within +/- this
    safety_margin_ml: float = 0.05     # extra lead so we stop slightly early
    stop_latency_s: float = 0.25       # pump + plumbing lag used for prediction
    min_confidence: float = 0.25       # ignore readings below this confidence
    require_stable_frames: int = 2     # consecutive at/over target before latch
    stream: bool = True                # continuously stream volume to the pump
    stream_min_interval_s: float = 0.1 # rate-limit streaming
    # --- command strings (edit to match your pump firmware) ---
    cmd_run: str = "RUN\n"
    cmd_stop: str = "STOP\n"
    cmd_setpoint: Optional[str] = None  # e.g. "SET:{target:.2f}\n"; None disables
    stream_format: str = "V:{volume:.2f}\n"
    answer_queries: bool = True         # honour legacy print/autoPrint/stopPrint


@dataclass
class ControlOutput:
    state: PumpState
    command: Optional[str]             # command string sent this cycle, if any
    streamed: bool                     # whether a stream line was sent
    at_target: bool
    remaining_ml: float
    predicted_stop_ml: float           # volume threshold at which we cut off


class PumpController:
    """
    Drives the pump toward a target volume with predictive, latching control.

    Typical use (per frame)::

        out = controller.update(reading.volume_ml, reading.flow_ml_per_s,
                                 reading.confidence, timestamp)

    where ``reading`` comes from LevelEstimator.update().
    """

    def __init__(self, link: PumpLink, cfg: Optional[PumpConfig] = None):
        self.link = link
        self.cfg = cfg or PumpConfig()
        self.state = PumpState.IDLE
        self._over_count = 0
        self._last_stream_t = 0.0
        self._auto_query = False
        self._last_volume = 0.0

    # --- target management ---
    def set_target(self, target_ml: float, direction: Optional[Direction] = None):
        self.cfg.target_ml = float(target_ml)
        if direction is not None:
            self.cfg.direction = direction
        self._over_count = 0
        self.state = PumpState.RUNNING
        if self.cfg.cmd_setpoint:
            self.link.write_line(self.cfg.cmd_setpoint.format(target=target_ml))

    def start(self):
        if self.state in (PumpState.IDLE, PumpState.PAUSED, PumpState.REACHED):
            self._over_count = 0
            self.state = PumpState.RUNNING

    def pause(self):
        self.state = PumpState.PAUSED
        self.link.write_line(self.cfg.cmd_stop)

    def _reached(self, volume: float) -> bool:
        cfg = self.cfg
        if cfg.direction == Direction.FILL:
            return volume >= cfg.target_ml - cfg.tolerance_ml
        return volume <= cfg.target_ml + cfg.tolerance_ml

    def _predicted_threshold(self, flow: float) -> float:
        """Volume at which to cut the pump so momentum lands us on target."""
        cfg = self.cfg
        lead = abs(flow) * cfg.stop_latency_s + cfg.safety_margin_ml
        if cfg.direction == Direction.FILL:
            return cfg.target_ml - lead
        return cfg.target_ml + lead

    def _should_stop(self, volume: float, flow: float) -> bool:
        thr = self._predicted_threshold(flow)
        if self.cfg.direction == Direction.FILL:
            return volume >= thr
        return volume <= thr

    def update(self, volume_ml: float, flow_ml_per_s: float,
               confidence: float, timestamp: Optional[float] = None) -> ControlOutput:
        cfg = self.cfg
        if timestamp is None:
            timestamp = time.time()

        # Handle any inbound legacy text commands first.
        self._handle_queries(volume_ml)

        streamed = False
        if cfg.stream and (timestamp - self._last_stream_t) >= cfg.stream_min_interval_s:
            self.link.write_line(cfg.stream_format.format(volume=volume_ml))
            self._last_stream_t = timestamp
            streamed = True

        command = None
        usable = confidence >= cfg.min_confidence
        if usable:
            self._last_volume = volume_ml

        predicted = self._predicted_threshold(flow_ml_per_s)

        if self.state == PumpState.RUNNING:
            if usable and self._should_stop(volume_ml, flow_ml_per_s):
                self._over_count += 1
                if self._over_count >= cfg.require_stable_frames:
                    self.link.write_line(cfg.cmd_stop)
                    command = cfg.cmd_stop
                    self.state = PumpState.REACHED
            else:
                self._over_count = 0
                # Keep the pump running (idempotent RUN each cycle is fine and
                # also acts as a watchdog heartbeat for many pump firmwares).
                self.link.write_line(cfg.cmd_run)
                command = cfg.cmd_run

        at_target = self._reached(self._last_volume)
        remaining = (cfg.target_ml - self._last_volume) if cfg.direction == Direction.FILL \
            else (self._last_volume - cfg.target_ml)

        return ControlOutput(
            state=self.state, command=command, streamed=streamed,
            at_target=at_target, remaining_ml=remaining,
            predicted_stop_ml=predicted,
        )

    def _handle_queries(self, volume_ml: float):
        """Backward-compatible text protocol from the previous firmware."""
        if not self.cfg.answer_queries:
            return
        incoming = self.link.read_available()
        if not incoming:
            if self._auto_query:
                self.link.write_line(f"{volume_ml:.2f}mL\n")
            return
        low = incoming.lower()
        if "autoprint" in low:
            self._auto_query = True
        elif "stopprint" in low:
            self._auto_query = False
        elif "print" in low:
            self.link.write_line(f"{volume_ml:.2f}mL\n")
