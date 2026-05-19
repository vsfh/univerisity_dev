import argparse
import os
import re
import shlex
import smtplib
import subprocess
import time
from dataclasses import dataclass
from email.message import EmailMessage
from pathlib import Path
from typing import List, Optional


# --- Configuration ---
DEFAULT_THRESHOLD_C = 49.0
DEFAULT_CHECK_INTERVAL = 30.0
DEFAULT_COOLDOWN = 1800.0
DEFAULT_SENSOR_PATTERN = r"MB_Air_Inlet|inlet|intake|ambient|system"
DEFAULT_SENSOR_NAME = "MB_Air_Inlet"


@dataclass
class TemperatureReading:
    source: str
    temperature_c: float


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore").strip()


def _read_hwmon_temperatures(sensor_pattern: str) -> List[TemperatureReading]:
    pattern = re.compile(sensor_pattern, re.IGNORECASE)
    readings: List[TemperatureReading] = []

    for hwmon_dir in sorted(Path("/sys/class/hwmon").glob("hwmon*")):
        chip_name = _read_text(hwmon_dir / "name") if (hwmon_dir / "name").exists() else hwmon_dir.name
        for input_path in sorted(hwmon_dir.glob("temp*_input")):
            idx = input_path.name.removeprefix("temp").removesuffix("_input")
            label_path = hwmon_dir / f"temp{idx}_label"
            label = _read_text(label_path) if label_path.exists() else f"temp{idx}"
            source = f"hwmon:{chip_name}:{label}"
            if not pattern.search(source):
                continue
            raw = _read_text(input_path)
            if not raw:
                continue
            value = float(raw)
            readings.append(TemperatureReading(source, value / 1000.0 if value > 200 else value))

    return readings


def _read_ipmitool_temperatures(sensor_pattern: str) -> List[TemperatureReading]:
    pattern = re.compile(sensor_pattern, re.IGNORECASE)
    try:
        proc = subprocess.run(
            ["ipmitool", "sensor"],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        return []

    if proc.returncode != 0:
        return []

    readings: List[TemperatureReading] = []
    for line in proc.stdout.splitlines():
        parts = [part.strip() for part in line.split("|")]
        if len(parts) < 3:
            continue
        name, value, unit = parts[0], parts[1], parts[2]
        source = f"ipmi:{name}"
        if not pattern.search(source) or "degrees C" not in unit:
            continue
        try:
            readings.append(TemperatureReading(source, float(value)))
        except ValueError:
            continue
    return readings


def _read_ipmitool_sensor_get(
    sensor_name: str,
    host: Optional[str],
    user: Optional[str],
    password: Optional[str],
) -> Optional[TemperatureReading]:
    command = ["ipmitool"]
    if host:
        if not user:
            raise RuntimeError("--ipmi-user is required when --ipmi-host is set.")
        if password is None:
            password = os.environ.get("IPMI_PASSWORD")
        if not password:
            raise RuntimeError("--ipmi-password or IPMI_PASSWORD is required when --ipmi-host is set.")
        command.extend(["-I", "lanplus", "-H", host, "-U", user, "-P", password])
    command.extend(["sensor", "get", sensor_name])

    try:
        proc = subprocess.run(
            command,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        return None

    if proc.returncode != 0:
        return None

    for line in proc.stdout.splitlines():
        if "Sensor Reading" not in line or "degrees C" not in line:
            continue
        match = re.search(r"([-+]?\d+(?:\.\d+)?)", line)
        if match is not None:
            return TemperatureReading(f"ipmi:{sensor_name}", float(match.group(1)))
    return None


def _read_command_temperature(command: str) -> TemperatureReading:
    proc = subprocess.run(
        shlex.split(command),
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    match = re.search(r"[-+]?\d+(?:\.\d+)?", proc.stdout)
    if match is None:
        raise ValueError(f"Sensor command did not print a numeric temperature: {command}")
    return TemperatureReading(f"command:{command}", float(match.group(0)))


def read_temperature(args: argparse.Namespace) -> TemperatureReading:
    if args.sensor_command:
        return _read_command_temperature(args.sensor_command)

    sensor_get = _read_ipmitool_sensor_get(
        sensor_name=args.sensor_name,
        host=args.ipmi_host,
        user=args.ipmi_user,
        password=args.ipmi_password,
    )
    if sensor_get is not None:
        return sensor_get

    readings = _read_hwmon_temperatures(args.sensor_pattern)
    if not readings:
        readings = _read_ipmitool_temperatures(args.sensor_pattern)
    if not readings:
        raise RuntimeError(
            "No inlet temperature sensor found. "
            "Pass --sensor-command or adjust --sensor-pattern."
        )
    return max(readings, key=lambda item: item.temperature_c)


def _smtp_port() -> int:
    return int(os.environ.get("ALERT_SMTP_PORT", "587"))


def _smtp_use_ssl() -> bool:
    return os.environ.get("ALERT_SMTP_SSL", "0").lower() in {"1", "true", "yes"}


def send_email(to_addr: str, subject: str, body: str) -> None:
    smtp_host = os.environ.get("ALERT_SMTP_HOST")
    from_addr = os.environ.get("ALERT_FROM")
    smtp_user = os.environ.get("ALERT_SMTP_USER")
    smtp_pass = os.environ.get("ALERT_SMTP_PASS")
    if not smtp_host:
        raise RuntimeError("ALERT_SMTP_HOST is required.")
    if not from_addr:
        raise RuntimeError("ALERT_FROM is required.")

    message = EmailMessage()
    message["From"] = from_addr
    message["To"] = to_addr
    message["Subject"] = subject
    message.set_content(body)

    client_cls = smtplib.SMTP_SSL if _smtp_use_ssl() else smtplib.SMTP
    client = client_cls(smtp_host, _smtp_port(), timeout=20)
    try:
        client.ehlo()
        if not _smtp_use_ssl():
            client.starttls()
            client.ehlo()
        if smtp_user or smtp_pass:
            client.login(smtp_user or from_addr, smtp_pass or "")
        client.send_message(message)
    finally:
        client.quit()


def run_monitor(args: argparse.Namespace) -> None:
    last_alert_time = 0.0
    while True:
        now = time.time()
        try:
            reading = read_temperature(args)
            line = (
                f"{time.strftime('%Y-%m-%d %H:%M:%S')} "
                f"{reading.source} {reading.temperature_c:.1f}C"
            )
            print(line, flush=True)
            if reading.temperature_c >= args.threshold and now - last_alert_time >= args.cooldown:
                send_email(
                    args.email_to,
                    f"[server] MB_Air_Inlet alert: {reading.temperature_c:.1f}C",
                    (
                        "Inlet temperature is above threshold.\n\n"
                        f"Source: {reading.source}\n"
                        f"Temperature: {reading.temperature_c:.1f}C\n"
                        f"Threshold: {args.threshold:.1f}C\n"
                        f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                    ),
                )
                last_alert_time = now
        except Exception as exc:
            print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} ERROR: {exc}", flush=True)
        time.sleep(args.interval)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Email alert for MB_Air_Inlet temperature.")
    parser.add_argument("--email-to", required=True)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD_C)
    parser.add_argument("--interval", type=float, default=DEFAULT_CHECK_INTERVAL)
    parser.add_argument("--cooldown", type=float, default=DEFAULT_COOLDOWN)
    parser.add_argument("--sensor-pattern", type=str, default=DEFAULT_SENSOR_PATTERN)
    parser.add_argument("--sensor-name", type=str, default=DEFAULT_SENSOR_NAME)
    parser.add_argument("--ipmi-host", type=str, default=None)
    parser.add_argument("--ipmi-user", type=str, default=None)
    parser.add_argument("--ipmi-password", type=str, default=None)
    parser.add_argument(
        "--sensor-command",
        type=str,
        default=None,
        help="Command that prints MB_Air_Inlet temperature in Celsius.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_monitor(parse_args())
