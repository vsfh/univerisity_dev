import os
import xml.etree.ElementTree as ET
from pathlib import Path

NAMESPACES = {
    "ns0": "http://www.opengis.net/kml/2.2",
    "ns1": "http://www.google.com/kml/ext/2.2",
}


def parse_kml(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    return tree, root


def extract_coordinates(root):
    coords_elem = root.find(".//ns0:coordinates", NAMESPACES)
    if coords_elem is None:
        coords_elem = root.find(".//coordinates")
    if coords_elem is not None and coords_elem.text:
        parts = coords_elem.text.strip().split(",")
        lon = float(parts[0])
        lat = float(parts[1])
        alt = float(parts[2]) if len(parts) > 2 else 0.0
        return lon, lat, alt
    return None, None, None


def extract_lookat(root):
    lookat = root.find(".//ns0:LookAt", NAMESPACES)
    if lookat is None:
        lookat = root.find(".//LookAt")
    if lookat is not None:
        longitude = (
            float(lookat.find("ns0:longitude", NAMESPACES).text)
            if lookat.find("ns0:longitude", NAMESPACES) is not None
            else 0.0
        )
        latitude = (
            float(lookat.find("ns0:latitude", NAMESPACES).text)
            if lookat.find("ns0:latitude", NAMESPACES) is not None
            else 0.0
        )
        altitude = (
            float(lookat.find("ns0:altitude", NAMESPACES).text)
            if lookat.find("ns0:altitude", NAMESPACES) is not None
            else 0.0
        )
        heading = (
            float(lookat.find("ns0:heading", NAMESPACES).text)
            if lookat.find("ns0:heading", NAMESPACES) is not None
            else 0.0
        )
        tilt = (
            float(lookat.find("ns0:tilt", NAMESPACES).text)
            if lookat.find("ns0:tilt", NAMESPACES) is not None
            else 0.0
        )
        range_val = (
            float(lookat.find("ns0:range", NAMESPACES).text)
            if lookat.find("ns0:range", NAMESPACES) is not None
            else 0.0
        )
        return {
            "longitude": longitude,
            "latitude": latitude,
            "altitude": altitude,
            "heading": heading,
            "tilt": tilt,
            "range": range_val,
        }
    return None


def update_lookat(root, lon, lat, alt, heading, tilt, range_val):
    lookat = root.find(".//ns0:LookAt", NAMESPACES)
    if lookat is None:
        lookat = root.find(".//LookAt")
    if lookat is not None:

        def set_text(elem, tag, value):
            target = elem.find(f"ns0:{tag}", NAMESPACES)
            if target is None:
                target = elem.find(tag)
            if target is not None:
                target.text = str(value)

        set_text(lookat, "longitude", lon)
        set_text(lookat, "latitude", lat)
        set_text(lookat, "altitude", alt)
        set_text(lookat, "heading", heading)
        set_text(lookat, "tilt", tilt)
        set_text(lookat, "range", range_val)


def update_coordinates(root, lon, lat, alt):
    coords_elem = root.find(".//ns0:coordinates", NAMESPACES)
    if coords_elem is None:
        coords_elem = root.find(".//coordinates")
    if coords_elem is not None:
        coords_elem.text = f"{lon},{lat},{alt}"


def generate_kml_files(input_folder, output_folder=None):
    input_path = Path(input_folder)
    if output_folder is None:
        output_folder = input_path / "generated"
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    range_values = [150, 200, 250, 300]
    heading_values = [0, 90, 180, 270]
    tilt_value = 45

    kml_files = list(input_path.glob("*.kml"))
    print(f"Found {len(kml_files)} KML files in {input_folder}")

    for kml_file in kml_files:
        print(f"Processing: {kml_file.name}")
        tree, root = parse_kml(kml_file)

        lon, lat, alt = extract_coordinates(root)
        if lon is None:
            print(f"  Warning: Could not extract coordinates from {kml_file.name}")
            continue

        lookat_data = extract_lookat(root)
        if lookat_data is None:
            print(f"  Warning: Could not extract LookAt from {kml_file.name}")
            continue

        original_name = root.find(".//ns0:name", NAMESPACES)
        if original_name is None:
            original_name = root.find(".//name")
        original_name_text = (
            original_name.text if original_name is not None else kml_file.stem
        )

        base_name = kml_file.stem

        for range_val in range_values:
            for heading_val in heading_values:
                tree_copy, root_copy = parse_kml(kml_file)

                update_lookat(
                    root_copy, lon, lat, alt, heading_val, tilt_value, range_val
                )
                update_coordinates(root_copy, lon, lat, alt)

                new_name = f"{base_name}_range{range_val}_heading{heading_val}.kml"
                output_file = output_path / new_name
                tree_copy.write(output_file, encoding="utf-8", xml_declaration=True)
                print(f"  Generated: {new_name}")

    print(f"\nAll generated files saved to: {output_folder}")


if __name__ == "__main__":
    import sys

    input_folder = '/data/feihong/kml_1024'
    output_folder = '/data/feihong/kml_drone_3_range_4_heading'
    generate_kml_files(input_folder, output_folder)
