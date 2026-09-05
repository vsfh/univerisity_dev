#!/usr/bin/env python3
"""Build an ID-to-location JSON file from KML point coordinates.

The default invocation processes both project KML directories and writes:

    {
      "0000": {
        "continent": "North America",
        "country": "Canada",
        "city": "Edmonton"
      }
    }

City/country names come from Nominatim reverse geocoding. Requests are
single-threaded, rate-limited, retried, and cached as JSON Lines, making a
stopped run safe to restart. The bundled Natural Earth country boundaries are
used as a fallback for country and continent names.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import requests


# --- Configuration ---
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIRS = (
    Path("/media/data1/feihong/kml_test_2"),
    Path("/media/data1/feihong/train_kml_2048"),
)
DEFAULT_OUTPUT = Path(__file__).resolve().with_name("kml_locations.json")
DEFAULT_CACHE = Path(__file__).resolve().with_name("kml_reverse_geocode_cache.jsonl")
DEFAULT_COUNTRY_GEOJSON = (
    REPO_ROOT / "tools/assets/ne_110m_admin_0_countries.geojson"
)
DEFAULT_ENDPOINT = "https://nominatim.openstreetmap.org/reverse"
DEFAULT_USER_AGENT = "university-kml-location-builder/1.0"
DEFAULT_MIN_DELAY = 1.05
DEFAULT_TIMEOUT = 30.0
DEFAULT_MAX_RETRIES = 5

CITY_FIELDS = (
    "city",
    "town",
    "village",
    "municipality",
    "borough",
    "hamlet",
    "suburb",
    "quarter",
    "city_district",
    "district",
    "county",
    "state_district",
    "state",
)

CONTINENT_ZH = {
    "Africa": "非洲",
    "Antarctica": "南极洲",
    "Asia": "亚洲",
    "Europe": "欧洲",
    "North America": "北美洲",
    "Oceania": "大洋洲",
    "South America": "南美洲",
    "Seven seas (open ocean)": "公海",
}


@dataclass(frozen=True)
class KmlLocation:
    location_id: str
    longitude: float
    latitude: float
    path: Path


@dataclass(frozen=True)
class CountryRecord:
    country_code: str
    country: str
    country_zh: str
    continent: str
    geometry: Mapping[str, Any]


class GeocodingError(RuntimeError):
    """Raised when reverse geocoding fails after all retries."""


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def parse_location_id(path: Path) -> str:
    match = re.search(r"(?<!\d)(\d{4})(?!\d)", path.stem)
    if match is None:
        raise ValueError(f"Filename does not contain a four-digit ID: {path}")
    return match.group(1)


def validate_coordinate(longitude: float, latitude: float, path: Path) -> Tuple[float, float]:
    if not (math.isfinite(longitude) and math.isfinite(latitude)):
        raise ValueError(f"Non-finite coordinate in {path}: {longitude}, {latitude}")
    if not (-180.0 <= longitude <= 180.0 and -90.0 <= latitude <= 90.0):
        raise ValueError(f"Out-of-range coordinate in {path}: {longitude}, {latitude}")
    return longitude, latitude


def parse_kml_coordinate(path: Path) -> Tuple[float, float]:
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as exc:
        raise ValueError(f"Invalid KML XML in {path}: {exc}") from exc

    for point in (element for element in root.iter() if local_name(element.tag) == "Point"):
        for child in point.iter():
            if local_name(child.tag) != "coordinates" or not child.text:
                continue
            token = child.text.strip().split()[0]
            parts = token.split(",")
            if len(parts) >= 2:
                return validate_coordinate(float(parts[0]), float(parts[1]), path)

    longitude: Optional[float] = None
    latitude: Optional[float] = None
    for element in root.iter():
        name = local_name(element.tag)
        if name == "longitude" and element.text and longitude is None:
            longitude = float(element.text.strip())
        elif name == "latitude" and element.text and latitude is None:
            latitude = float(element.text.strip())
        if longitude is not None and latitude is not None:
            return validate_coordinate(longitude, latitude, path)

    raise ValueError(f"No Point or LookAt coordinate found in {path}")


def discover_locations(input_dirs: Sequence[Path]) -> List[KmlLocation]:
    by_id: Dict[str, KmlLocation] = {}
    for input_dir in input_dirs:
        if not input_dir.is_dir():
            raise NotADirectoryError(f"KML directory does not exist: {input_dir}")
        for path in sorted(input_dir.glob("*.kml"), key=lambda item: item.name):
            location_id = parse_location_id(path)
            if location_id in by_id:
                previous = by_id[location_id].path
                raise ValueError(
                    f"Duplicate four-digit ID {location_id}: {previous} and {path}"
                )
            longitude, latitude = parse_kml_coordinate(path)
            by_id[location_id] = KmlLocation(
                location_id=location_id,
                longitude=longitude,
                latitude=latitude,
                path=path,
            )
    return [by_id[key] for key in sorted(by_id)]


def iter_polygons(geometry: Mapping[str, Any]) -> Iterable[Sequence[Sequence[Sequence[float]]]]:
    geometry_type = geometry.get("type")
    coordinates = geometry.get("coordinates", [])
    if geometry_type == "Polygon":
        yield coordinates
    elif geometry_type == "MultiPolygon":
        yield from coordinates


def point_in_ring(longitude: float, latitude: float, ring: Sequence[Sequence[float]]) -> bool:
    inside = False
    if len(ring) < 3:
        return False
    previous = ring[-1]
    for current in ring:
        x1, y1 = float(previous[0]), float(previous[1])
        x2, y2 = float(current[0]), float(current[1])
        crosses = (y1 > latitude) != (y2 > latitude)
        if crosses:
            intersection = (x2 - x1) * (latitude - y1) / (y2 - y1) + x1
            if longitude < intersection:
                inside = not inside
        previous = current
    return inside


def point_in_polygon(
    longitude: float,
    latitude: float,
    polygon: Sequence[Sequence[Sequence[float]]],
) -> bool:
    if not polygon or not point_in_ring(longitude, latitude, polygon[0]):
        return False
    return not any(point_in_ring(longitude, latitude, hole) for hole in polygon[1:])


class CountryIndex:
    def __init__(self, geojson_path: Path) -> None:
        payload = json.loads(geojson_path.read_text(encoding="utf-8"))
        self.records: List[CountryRecord] = []
        self.by_code: Dict[str, CountryRecord] = {}
        for feature in payload.get("features", []):
            properties = feature.get("properties", {})
            geometry = feature.get("geometry")
            if not isinstance(geometry, dict):
                continue
            code = str(properties.get("ISO_A2") or "").upper()
            record = CountryRecord(
                country_code=code,
                country=str(properties.get("NAME_EN") or properties.get("ADMIN") or ""),
                country_zh=str(properties.get("NAME_ZH") or ""),
                continent=str(properties.get("CONTINENT") or ""),
                geometry=geometry,
            )
            self.records.append(record)
            if len(code) == 2 and code != "-99":
                self.by_code[code] = record

    def lookup(
        self,
        country_code: str,
        longitude: float,
        latitude: float,
    ) -> Optional[CountryRecord]:
        by_code = self.by_code.get(country_code.upper())
        if by_code is not None:
            return by_code
        for record in self.records:
            if any(
                point_in_polygon(longitude, latitude, polygon)
                for polygon in iter_polygons(record.geometry)
            ):
                return record
        return None


def cache_key(location: KmlLocation, language: str) -> str:
    return f"{language}|{location.latitude:.7f}|{location.longitude:.7f}"


def load_cache(path: Path) -> Dict[str, Dict[str, str]]:
    cache: Dict[str, Dict[str, str]] = {}
    if not path.exists():
        return cache
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                key = str(record["key"])
                value = record["value"]
                if not isinstance(value, dict):
                    raise TypeError("value is not an object")
                cache[key] = {
                    "continent": str(value.get("continent") or ""),
                    "country": str(value.get("country") or ""),
                    "city": str(value.get("city") or ""),
                }
            except (KeyError, TypeError, json.JSONDecodeError) as exc:
                print(
                    f"Warning: ignoring invalid cache line {line_number} in {path}: {exc}",
                    file=sys.stderr,
                )
    return cache


def append_cache(path: Path, key: str, value: Mapping[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"key": key, "value": dict(value)}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def localized_fallback(record: Optional[CountryRecord], language: str) -> Tuple[str, str]:
    if record is None:
        return "", ""
    use_chinese = language.lower().startswith("zh")
    continent = CONTINENT_ZH.get(record.continent, record.continent) if use_chinese else record.continent
    country = record.country_zh if use_chinese and record.country_zh else record.country
    return continent, country


def normalize_response(
    payload: Mapping[str, Any],
    location: KmlLocation,
    country_index: CountryIndex,
    language: str,
) -> Dict[str, str]:
    address_value = payload.get("address", {})
    address = address_value if isinstance(address_value, dict) else {}
    country_code = str(address.get("country_code") or "").upper()
    fallback = country_index.lookup(country_code, location.longitude, location.latitude)
    fallback_continent, fallback_country = localized_fallback(fallback, language)

    continent = str(address.get("continent") or fallback_continent)
    country = str(address.get("country") or fallback_country)
    city = next((str(address[key]) for key in CITY_FIELDS if address.get(key)), "")

    # Dict insertion order is intentional and becomes the output JSON field order.
    return {
        "continent": continent,
        "country": country,
        "city": city,
    }


class NominatimClient:
    def __init__(
        self,
        endpoint: str,
        user_agent: str,
        email: Optional[str],
        language: str,
        min_delay: float,
        timeout: float,
        max_retries: int,
    ) -> None:
        self.endpoint = endpoint
        self.email = email
        self.language = language
        self.min_delay = min_delay
        self.timeout = timeout
        self.max_retries = max_retries
        self.last_request_time = 0.0
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": user_agent,
                "Accept": "application/json",
            }
        )

    def reverse(self, latitude: float, longitude: float) -> Mapping[str, Any]:
        params: Dict[str, Any] = {
            "lat": f"{latitude:.8f}",
            "lon": f"{longitude:.8f}",
            "format": "jsonv2",
            "addressdetails": 1,
            "zoom": 10,
            "accept-language": self.language,
        }
        if self.email:
            params["email"] = self.email

        last_error = "unknown error"
        for attempt in range(self.max_retries + 1):
            elapsed = time.monotonic() - self.last_request_time
            if elapsed < self.min_delay:
                time.sleep(self.min_delay - elapsed)
            try:
                response = self.session.get(
                    self.endpoint,
                    params=params,
                    timeout=self.timeout,
                )
                self.last_request_time = time.monotonic()
                if response.status_code == 429:
                    retry_after = response.headers.get("Retry-After", "")
                    try:
                        wait_seconds = max(float(retry_after), self.min_delay)
                    except ValueError:
                        wait_seconds = min(2.0 ** (attempt + 1), 60.0)
                    last_error = f"HTTP 429; retry after {wait_seconds:.1f}s"
                    time.sleep(wait_seconds)
                    continue
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, dict):
                    raise ValueError("response JSON is not an object")
                if payload.get("error"):
                    raise ValueError(str(payload["error"]))
                return payload
            except (requests.RequestException, ValueError) as exc:
                last_error = str(exc)
                if attempt < self.max_retries:
                    time.sleep(min(2.0 ** (attempt + 1), 60.0))

        raise GeocodingError(
            f"Reverse geocoding failed for ({latitude}, {longitude}): {last_error}"
        )


def write_output(path: Path, results: Mapping[str, Mapping[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = {key: dict(results[key]) for key in sorted(results)}
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(ordered, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch reverse-geocode point coordinates from KML files."
    )
    parser.add_argument(
        "--input-dirs",
        nargs="+",
        type=Path,
        default=list(DEFAULT_INPUT_DIRS),
        help="Directories containing KML files.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--country-geojson", type=Path, default=DEFAULT_COUNTRY_GEOJSON)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--language", default="en", help="Nominatim result language, e.g. en or zh-CN.")
    parser.add_argument("--user-agent", default=DEFAULT_USER_AGENT)
    parser.add_argument("--email", help="Optional contact email sent to Nominatim.")
    parser.add_argument("--min-delay", type=float, default=DEFAULT_MIN_DELAY)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    parser.add_argument(
        "--max-consecutive-failures",
        type=int,
        default=3,
        help="Stop early if this many coordinates fail consecutively.",
    )
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--limit", type=int, help="Process only the first N IDs.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and validate inputs without making HTTP requests.",
    )
    args = parser.parse_args()
    if args.min_delay < 1.0 and args.endpoint == DEFAULT_ENDPOINT:
        parser.error("The public Nominatim endpoint requires --min-delay >= 1.0 seconds.")
    if args.max_retries < 0:
        parser.error("--max-retries must be non-negative.")
    if args.max_consecutive_failures <= 0:
        parser.error("--max-consecutive-failures must be positive.")
    if args.save_every <= 0:
        parser.error("--save-every must be positive.")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive.")
    return args


def main() -> int:
    args = parse_args()
    locations = discover_locations(args.input_dirs)
    if args.limit is not None:
        locations = locations[: args.limit]
    print(f"Discovered {len(locations)} unique KML coordinates.")
    if args.dry_run:
        print("Dry run completed; no network requests or output files were created.")
        return 0

    country_index = CountryIndex(args.country_geojson)
    cache = load_cache(args.cache)
    client = NominatimClient(
        endpoint=args.endpoint,
        user_agent=args.user_agent,
        email=args.email,
        language=args.language,
        min_delay=args.min_delay,
        timeout=args.timeout,
        max_retries=args.max_retries,
    )

    results: Dict[str, Dict[str, str]] = {}
    failures: List[str] = []
    consecutive_failures = 0
    cache_hits = 0

    try:
        for index, location in enumerate(locations, start=1):
            key = cache_key(location, args.language)
            value = cache.get(key)
            if value is not None:
                cache_hits += 1
            else:
                try:
                    payload = client.reverse(location.latitude, location.longitude)
                    value = normalize_response(
                        payload,
                        location,
                        country_index,
                        args.language,
                    )
                    cache[key] = value
                    append_cache(args.cache, key, value)
                    consecutive_failures = 0
                except GeocodingError as exc:
                    failures.append(f"{location.location_id}: {exc}")
                    consecutive_failures += 1
                    print(f"Error: {failures[-1]}", file=sys.stderr)
                    if consecutive_failures >= args.max_consecutive_failures:
                        print(
                            "Stopping after consecutive geocoding failures; rerun to resume.",
                            file=sys.stderr,
                        )
                        break
                    continue

            results[location.location_id] = value
            if index % args.save_every == 0 or index == len(locations):
                write_output(args.output, results)
                print(
                    f"[{index}/{len(locations)}] saved {len(results)} locations "
                    f"({cache_hits} cache hits) to {args.output}"
                )
    except KeyboardInterrupt:
        print("Interrupted; saving completed results before exit.", file=sys.stderr)
        write_output(args.output, results)
        return 130

    write_output(args.output, results)
    missing_fields = {
        field: sum(not value.get(field) for value in results.values())
        for field in ("continent", "country", "city")
    }
    print(
        f"Finished: {len(results)}/{len(locations)} locations written to {args.output}. "
        f"Missing fields: {missing_fields}."
    )
    if failures or len(results) != len(locations):
        print(f"Unresolved requests: {len(failures)}. Rerun the same command to resume.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
