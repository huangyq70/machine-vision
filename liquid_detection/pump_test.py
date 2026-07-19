r"""
pump_test.py -- Verify serial/USB communication with your pump.

Use this BEFORE running the monitor to confirm the Pi can actually talk to the
pump, discover which /dev/tty* device it is, and figure out the exact command
strings your pump firmware expects (which you then put into PumpConfig in
pump_controller.py).

List candidate serial ports:
    python pump_test.py --scan

Open an interactive console (type a line, it is sent with a newline; anything
the pump sends back is printed):
    python pump_test.py --port /dev/ttyACM0 --baud 9600

    > RUN            # sends "RUN\n"
    > STOP           # sends "STOP\n"
    > \x02RUN\r      # escapes allowed: \r \n \t \xHH
    > :run           # (whatever your pump's manual says)
    quit             # exit

Send a canned burst and watch the reply (useful for one-off checks):
    python pump_test.py --port /dev/ttyACM0 --send "RUN" --wait 2 --send "STOP"
"""

from __future__ import annotations

import argparse
import sys
import time


def list_ports():
    try:
        from serial.tools import list_ports
    except Exception:
        print("pyserial not installed. Run: pip install pyserial")
        return
    ports = list(list_ports.comports())
    if not ports:
        print("No serial ports found. Is the pump plugged in and powered?")
        print("Tip: run 'dmesg | tail' right after plugging it in.")
        return
    print("Available serial ports:")
    for p in ports:
        print(f"  {p.device:20s} {p.description}  [{p.hwid}]")


def unescape(s: str) -> bytes:
    """Turn a typed string with \\r \\n \\t \\xHH escapes into bytes."""
    return s.encode("ascii", "ignore").decode("unicode_escape").encode("latin-1")


def open_port(port, baud):
    import serial
    return serial.Serial(port, baud, timeout=0.2)


def drain(ser, seconds):
    """Print everything received for the given number of seconds."""
    end = time.monotonic() + seconds
    got = bytearray()
    while time.monotonic() < end:
        n = ser.in_waiting if hasattr(ser, "in_waiting") else 0
        if n:
            got += ser.read(n)
        else:
            time.sleep(0.02)
    if got:
        print(f"  <- {got!r}")
    else:
        print("  <- (no reply)")


def main():
    ap = argparse.ArgumentParser(description="Pump serial comms tester")
    ap.add_argument("--scan", action="store_true", help="list serial ports and exit")
    ap.add_argument("--listen", action="store_true",
                    help="passively print everything the pump sends (read-only)")
    ap.add_argument("--port")
    ap.add_argument("--baud", type=int, default=9600)
    ap.add_argument("--send", action="append", default=[],
                    help="send this line (repeatable); newline appended")
    ap.add_argument("--wait", type=float, default=1.0,
                    help="seconds to listen after each --send")
    ap.add_argument("--no-newline", action="store_true",
                    help="do not append a newline to sent lines")
    args = ap.parse_args()

    if args.scan or not args.port:
        list_ports()
        if not args.port:
            print("\nRe-run with --port <device> to open a console.")
        return

    try:
        ser = open_port(args.port, args.baud)
    except Exception as e:  # noqa: BLE001
        print(f"Could not open {args.port}: {e}")
        print("Check the device path (--scan), the cable, and that your user is")
        print("in the 'dialout' group (sudo usermod -aG dialout $USER, then re-login).")
        sys.exit(1)

    term = "" if args.no_newline else "\n"

    # Passive listen: print everything the pump sends, send nothing.
    if args.listen:
        print(f"Listening on {args.port} @ {args.baud} (read-only). Ctrl-C to stop.")
        print("If the pump polls the Pi, you should see 'print' / 'autoPrint' here.")
        try:
            while True:
                n = ser.in_waiting if hasattr(ser, "in_waiting") else 0
                if n:
                    print(f"  <- {ser.read(n)!r}")
                else:
                    time.sleep(0.05)
        except KeyboardInterrupt:
            pass
        finally:
            ser.close()
            print("\nClosed.")
        return

    print(f"Opened {args.port} @ {args.baud}. Ctrl-C or 'quit' to exit.")

    # Non-interactive canned bursts.
    if args.send:
        for line in args.send:
            payload = unescape(line) + term.encode()
            print(f"  -> {payload!r}")
            ser.write(payload)
            drain(ser, args.wait)
        ser.close()
        return

    # Interactive console.
    try:
        while True:
            try:
                line = input("> ")
            except EOFError:
                break
            if line.strip().lower() in ("quit", "exit"):
                break
            payload = unescape(line) + term.encode()
            print(f"  -> {payload!r}")
            ser.write(payload)
            drain(ser, args.wait)
    except KeyboardInterrupt:
        pass
    finally:
        ser.close()
        print("\nClosed.")


if __name__ == "__main__":
    main()
