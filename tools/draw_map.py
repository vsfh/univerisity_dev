import matplotlib.pyplot as plt
import os
import requests
import numpy as np
import time
import math
import tempfile
import shutil
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable
from typing import List, Dict, Tuple

def get_world_map_two():
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    input_folder = "/data/feihong/kml_1024"
    kml_files = [
        os.path.join(input_folder, f)
        for f in os.listdir(input_folder)
        if f.lower().endswith(".kml")
    ]

    print(f"找到 {len(kml_files)} 个KML文件需要处理")

    # Sample data: Lists of latitudes and longitudes
    lats = []
    lons = []
    for kml_file in kml_files:
        input_path = os.path.join(input_folder, kml_file)
        lat, lon = process_kml_file(input_path)
        lats.append(lat)
        lons.append(lon)

    lons_arr = np.array(lons)
    lats_arr = np.array(lats)

    input_folder = "/data/feihong/new_univ_kml"
    kml_files = [
        os.path.join(input_folder, f)
        for f in os.listdir(input_folder)
        if f.lower().endswith(".kml")
    ]

    print(f"找到 {len(kml_files)} 个KML文件需要处理")

    # Sample data: Lists of latitudes and longitudes
    lats = []
    lons = []
    for kml_file in kml_files:
        input_path = os.path.join(input_folder, kml_file)
        lat, lon = process_kml_file(input_path)
        lats.append(lat)
        lons.append(lon)
    # 2. Spatial Aggregation (Grid-based Clustering)
    grid_size = 1.5
    # Combine lons and lats into a single 2D array
    df_coords = np.column_stack((lons_arr, lats_arr))

    # Round coordinates to the nearest grid intersection
    clustered_coords = np.round(df_coords / grid_size) * grid_size

    # Count unique clusters and their sizes
    unique_locs, counts = np.unique(clustered_coords, axis=0, return_counts=True)
    clons = unique_locs[:, 0]
    clats = unique_locs[:, 1]

    # 3. Plotting the Robinson Projection
    fig = plt.figure(figsize=(20, 10))
    ax = plt.axes(projection=ccrs.Robinson())

    # Professional Grey Basemap
    ax.add_feature(
        cfeature.LAND, facecolor="#efefef", edgecolor="#d1d1d1", linewidth=0.5
    )
    ax.add_feature(cfeature.OCEAN, facecolor="#ffffff")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.4, edgecolor="#aaaaaa")
    ax.set_global()

    # 4. Scaling the Circles (Area-based scaling)
    # Adjust the multiplier (25) based on your visual preference
    sizes = np.sqrt(counts) * 25

    scatter = ax.scatter(
        clons,
        clats,
        s=sizes,
        facecolor="#8e44ad",
        edgecolor="#6c3483",
        alpha=0.4,
        linewidth=0.7,
        transform=ccrs.PlateCarree(),
    )

    # 5. Legend (Manual scale)
    legend_counts = [10, 100, 500, 1000]
    for c in legend_counts:
        # Use the same scaling logic (np.sqrt(c) * 25) for legend accuracy
        plt.scatter(
            [],
            [],
            c="#8e44ad",
            alpha=0.4,
            s=np.sqrt(c) * 25,
            label=f"{c}",
            edgecolor="#6c3483",
        )

    plt.legend(
        scatterpoints=1,
        frameon=False,
        labelspacing=1.8,
        title="Building Count",
        loc="lower left",
        bbox_to_anchor=(0.02, 0.1),
    )

    plt.title(
        "Spatial Distribution and Density of Wild-University Landmarks",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )

    # Save as PDF for high-quality ECCV submission
    plt.savefig("WildUniversity_Clustered_Map.pdf", dpi=300, bbox_inches="tight")
    plt.show()


# --- Configuration ---
ASIAN_TOP_200 = [
    "The University of Tokyo",
    "Tokyo Institute of Technology",
    "Waseda University",
    "Keio University",
    "Hitotsubashi University",
    "Tokyo Medical and Dental University",
    "Tokyo University of Science",
    "Tokyo University of Agriculture and Technology",
    "Sophia University",
    "Osaka University",
    "Osaka Metropolitan University",
    "Kyoto University",
    "Ritsumeikan University",
    "Yokohama City University",
    "Kobe University",
    "Hiroshima University",
    "Nagoya University",
    "National Taiwan University",
    "National Taiwan University of Science and Technology",
    "National Taipei University of Technology",
    "National Taiwan Normal University",
    "National Chengchi University",
    "Taipei Medical University",
    "National Chung Hsing University",
    "China Medical University, Taiwan",
    "National Tsing Hua University",
    "National Yang Ming Chiao Tung University",
    "The University of Hong Kong",
    "The Chinese University of Hong Kong",
    "The Hong Kong University of Science and Technology",
    "The Hong Kong Polytechnic University",
    "City University of Hong Kong",
    "Hong Kong Baptist University",
    "Lingnan University (Hong Kong)",
    "University of Macau",
    "Macau University of Science and Technology",
    "National University of Singapore",
    "Nanyang Technological University, Singapore",
    "Singapore Management University",
    "Singapore University of Technology and Design",
    "Chulalongkorn University",
    "Mahidol University",
    "Thammasat University",
    "Kasetsart University",
    "King Mongkut's University of Technology Thonburi",
    "Universiti Malaya",
    "Universiti Putra Malaysia",
    "Universiti Kebangsaan Malaysia",
    "Taylor's University",
    "UCSI University",
    "Sunway University",
    "Management and Science University",
    "Universiti Teknologi MARA",
    "International Islamic University Malaysia",
    "Universiti Tenaga Nasional",
    "The University of Melbourne",
    "The University of Sydney",
    "UNSW Sydney",
    "Australian National University",
    "Monash University",
    "The University of Queensland",
    "The University of Western Australia",
    "The University of Adelaide",
    "University of Technology Sydney",
    "RMIT University",
    "Macquarie University",
    "Curtin University",
    "Deakin University",
    "Queensland University of Technology (QUT)",
    "La Trobe University",
    "Griffith University",
    "Swinburne University of Technology",
    "University of South Australia",
    "Western Sydney University",
    "Flinders University",
    "University of Canberra",
    "Murdoch University",
    "Edith Cowan University",
    "Southern Cross University",
    "Bond University",
    "Victoria University",
    "Australian Catholic University",
    "University of Auckland",
    "Massey University",
    "Victoria University of Wellington",
    "University of Canterbury",
    "Lincoln University",
    "Auckland University of Technology (AUT)",
    "Universidad de Buenos Aires (UBA)",
    "Pontificia Universidad Católica Argentina",
    "Universidad Austral",
    "Universidad de Belgrano",
    "Instituto Tecnológico de Buenos Aires (ITBA)",
    "Universidad de San Andrés",
    "Universidad Torcuato Di Tella",
    # 英国 (UK)
    "Imperial College London",
    "University College London",
    "University of Manchester",
    "University of Birmingham",
    "University of Leeds",
    "University of Glasgow",
    "University of Edinburgh",
    "University of Oxford",
    "University of Cambridge",
    # 法国 (France)
    "Université PSL",
    "Institut Polytechnique de Paris",
    "ENS de Lyon",
    "Université Claude Bernard Lyon 1",
    "Aix-Marseille University",
    "Université Côte d'Azur",
    "University of Bordeaux",
    "Université de Toulouse",
    "University of Strasbourg",
    # 荷兰 (Netherlands)
    "University of Amsterdam",
    "Vrije Universiteit Amsterdam",
    "Erasmus University Rotterdam",
    "Utrecht University",
    "Leiden University",
    "Eindhoven University of Technology",
    # 比利时 (Belgium)
    "Université Libre de Bruxelles",
    "Vrije Universiteit Brussel",
    "University of Antwerp",
    "Ghent University",
    # 北欧 (Northern Europe)
    "KTH Royal Institute of Technology",
    "Stockholm University",
    "University of Oslo",
    "University of Copenhagen",
    "Technical University of Denmark",
    "University of Helsinki",
    "Aalto University",
    # 德国 (Germany)
    "Freie Universität Berlin",
    "Humboldt-Universität zu Berlin",
    "Technical University of Munich",
    "Ludwig-Maximilians-Universität München",
    "University of Hamburg",
    "University of Cologne",
    "Goethe University Frankfurt",
    "University of Stuttgart",
    "TU Dortmund University",
    # 瑞士 (Switzerland)
    "ETH Zurich",
    "University of Zurich",
    "University of Geneva",
    "University of Basel",
    "University of Bern",
    # 奥地利 (Austria)
    "University of Vienna",
    "TU Wien",
    "Paris Lodron University of Salzburg",
    "University of Innsbruck",
    # 捷克 (Czech Republic)
    "Charles University",
    "Czech Technical University in Prague",
    "University of West Bohemia",
    # 波兰 (Poland)
    "University of Warsaw",
    "Warsaw University of Technology",
    "Jagiellonian University",
    "AGH University of Krakow",
    "University of Wroclaw",
    "Wrocław University of Science and Technology",
    "Gdańsk University of Technology",
    # 意大利 (Italy)
    "Sapienza University of Rome",
    "University of Rome Tor Vergata",
    "Politecnico di Milano",
    "University of Milan",
    "University of Florence",
    "Ca' Foscari University of Venice",
    "University of Naples - Federico II",
    "Politecnico di Torino",
    "University of Turin",
    "University of Genoa",
    # 西班牙 (Spain)
    "Universidad Autónoma de Madrid",
    "Complutense University of Madrid",
    "University of Barcelona",
    "Autonomous University of Barcelona",
    "University of Valencia",
    "Universitat Politècnica de València",
    "University of Seville",
    "University of Malaga",
    # 葡萄牙 (Portugal)
    "University of Lisbon",
    "Universidade NOVA de Lisboa",
    "University of Porto",
    # 希腊 (Greece)
    "National Technical University of Athens",
    "National and Kapodistrian University of Athens",
    "Aristotle University of Thessaloniki",
]


def get_lib_coord():
    import time
    geolocator = Nominatim(user_agent="asian_university_locator")

    output_file = "runs/school_name.txt"
    # output_file = "south_america_50_libraries_free.txt"

    def get_free_coordinates(university_name):
        """Tries to find the library first, then falls back to the main campus."""

        campus = geolocator.geocode(university_name, timeout=10)
        if not campus:
            return "Campus not found", None, None, "Failed at Step 1"

        campus_lat = campus.latitude
        campus_lon = campus.longitude

        # STEP 2: Use Overpass API to find the exact building polygon near that point
        # We search within an 800-meter radius (around:800) for anything tagged as a library
        overpass_url = "https://overpass-api.de/api/interpreter"

        overpass_query = f"""
            [out:json][timeout:25];
            (
            nwr["amenity"~"library"](around:1800, {campus_lat}, {campus_lon});
            nwr["building"~"library"](around:1800, {campus_lat}, {campus_lon});
            nwr["name"~"Library", i](around:1800, {campus_lat}, {campus_lon});
            );
            out center; 
            """

        try:
            response = requests.post(
                "https://overpass-api.de/api/interpreter", data={"data": overpass_query}
            )

            if response.status_code != 200:
                return (
                    f"{university_name} Campus Center",
                    campus_lat,
                    campus_lon,
                    "Fallback: Server Blocked",
                )

            data = response.json()

            if data and "elements" in data and len(data["elements"]) > 0:
                import math

                best_match = None
                closest_dist = float("inf")

                # Grab the first word of the university (e.g., "Peking" or "Tsinghua")
                # to check if it's in the library's name
                search_keyword = university_name.split()[0].lower()

                for element in data["elements"]:
                    # Overpass puts coords in 'center' for polygons, and direct lat/lon for nodes
                    lat = element.get("center", element).get("lat")
                    lon = element.get("center", element).get("lon")

                    tags = element.get("tags", {})
                    name = tags.get("name", "")
                    name_en = tags.get("name:en", "")

                    # Calculate rough distance from the campus center
                    dist = math.hypot(lat - campus_lat, lon - campus_lon)

                    # If the building's name explicitly contains "Peking" (or whatever the university is),
                    # we artificially drop the distance score so it is guaranteed to win.
                    if (
                        search_keyword in name.lower()
                        or search_keyword in name_en.lower()
                    ):
                        dist -= 1000

                    if dist < closest_dist:
                        closest_dist = dist
                        best_match = (
                            name or name_en or f"{university_name} Library",
                            lat,
                            lon,
                        )

                if best_match:
                    return (
                        best_match[0],
                        best_match[1],
                        best_match[2],
                        "Exact Polygon Center",
                    )

            # If the loop finds nothing, fallback
            return (
                f"{university_name} Campus Center",
                campus_lat,
                campus_lon,
                "Fallback to Campus",
            )

        except requests.exceptions.JSONDecodeError:
            return (
                f"{university_name} Campus Center",
                campus_lat,
                campus_lon,
                "Fallback: JSON Error",
            )
        except Exception as e:
            return "API Error", campus_lat, campus_lon, "Fallback to Campus"

    # Run the automation and save to a text file
    with open(output_file, "w", encoding="utf-8") as file:
        # Notice we added a "Search Used" column so you know if it grabbed the library or the campus
        file.write("University, Matched Location, Latitude, Longitude, Search Used\n")

        for school in ASIAN_TOP_200:
            print(f"Searching for: {school}...")
            res = get_free_coordinates(school)
            if res[-1] == "Exact Polygon Center":
                match_name, lat, lon, express = res
            else:
                continue

            file.write(f"{school}, {match_name}, {lat}, {lon}\n")

            # CRITICAL: OpenStreetMap is free, but they will temporarily block you
            # if you search too fast. A 1.5 second delay keeps you safe.
            time.sleep(1.5)

    print(f"\nSuccess! All coordinates saved to {output_file}")


def process_kml_file(input_path):
    import xml.etree.ElementTree as ET

    """处理单个KML文件"""
    # 获取文件名（不带扩展名）
    filename = os.path.splitext(os.path.basename(input_path))[0]

    # 解析KML文件
    tree = ET.parse(input_path)
    root = tree.getroot()

    # 定义命名空间（不同KML文件可能使用不同的命名空间）
    ns = {"kml": "http://www.opengis.net/kml/2.2"}

    # 修改name元素为文件名
    name_elem = root.find(".//kml:name", ns)
    if name_elem is not None:
        name_elem.text = filename

    # 处理所有Placemark的LookAt元素
    for lookat in root.findall(".//kml:LookAt", ns):
        longitude_elem = lookat.find("kml:longitude", ns)
        latitude_elem = lookat.find("kml:latitude", ns)

        # Extract text and convert to float
        if longitude_elem is not None and latitude_elem is not None:
            lon_float = float(longitude_elem.text)
            lat_float = float(latitude_elem.text)

    return lat_float, lon_float


def get_world_map():
    import geopandas as gpd

    input_folder = "/data/feihong/kml_1024"
    kml_files = [
        os.path.join(input_folder, f)
        for f in os.listdir(input_folder)
        if f.lower().endswith(".kml")
    ]

    print(f"找到 {len(kml_files)} 个KML文件需要处理")

    # Sample data: Lists of latitudes and longitudes
    lats = []
    lons = []
    for kml_file in kml_files:
        input_path = os.path.join(input_folder, kml_file)
        lat, lon = process_kml_file(input_path)
        lats.append(lat)
        lons.append(lon)

    # --- THE FIX ---
    # We now read the shapefile directly from the official Natural Earth URL
    url = (
        "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
    )
    world = gpd.read_file(url)

    # Create the plot figure and specify the size
    fig, ax = plt.subplots(figsize=(15, 10))

    # Draw the world map
    world.plot(ax=ax, color="lightgray", edgecolor="white")

    # Plot the coordinates as red scatter points
    # Matplotlib expects (X, Y) which translates to (Longitude, Latitude)
    ax.scatter(
        lons, lats, color="red", s=50, zorder=5, alpha=0.8, label="University-1652"
    )

    input_folder = "/data/feihong/new_univ_kml"
    kml_files = [
        os.path.join(input_folder, f)
        for f in os.listdir(input_folder)
        if f.lower().endswith(".kml")
    ]

    print(f"找到 {len(kml_files)} 个KML文件需要处理")

    # Sample data: Lists of latitudes and longitudes
    lats = []
    lons = []
    for kml_file in kml_files:
        input_path = os.path.join(input_folder, kml_file)
        lat, lon = process_kml_file(input_path)
        lats.append(lat)
        lons.append(lon)

    ax.scatter(
        lons, lats, color="green", s=50, zorder=5, alpha=0.8, label="Wild-University"
    )

    # Add a title
    # plt.title("Locations of University", fontsize=16)
    ax.legend(loc="lower right", fontsize=12, frameon=True, framealpha=1, title="")
    # Remove the axis ticks (makes it look cleaner)
    ax.set_xticks([])
    ax.set_yticks([])

    # Save the map as a PNG image
    output_image = "static_world_map.png"
    plt.savefig(output_image, bbox_inches="tight", dpi=300)

    print(f"Map saved successfully as '{output_image}'!")


def match_libraries():
    """Read library names from asian_200_libraries_free.txt and match with get_free_coordinates() results."""
    import time
    import math
    from geopy.geocoders import Nominatim
    from geopy.exc import GeocoderTimedOut

    library_file = "/data/feihong/univerisity_dev/asian_200_libraries_free.txt"
    output_file = "/data/feihong/univerisity_dev/matched_university_libraries.txt"

    def haversine_distance(lat1, lon1, lat2, lon2):
        R = 6371
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lat = math.radians(lat2 - lat1)
        delta_lon = math.radians(lon2 - lon1)

        a = (
            math.sin(delta_lat / 2) ** 2
            + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2) ** 2
        )
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return R * c

    geolocator = Nominatim(user_agent="university_matcher")

    print("Caching university coordinates...")
    university_coords = {}
    for i, university in enumerate(ASIAN_TOP_200):
        print(f"  [{i + 1}/{len(ASIAN_TOP_200)}] Geocoding: {university}")
        try:
            location = geolocator.geocode(university, timeout=10)
            if location:
                university_coords[university] = (location.latitude, location.longitude)
                print(f"    -> ({location.latitude:.4f}, {location.longitude:.4f})")
            else:
                print(f"    -> Not found")
        except Exception as e:
            print(f"    -> Error: {e}")
        time.sleep(1.1)

    print(f"\nSuccessfully geocoded {len(university_coords)} universities")

    print("\nMatching libraries to nearest universities...")
    with open(library_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    results = []
    for line_num, line in enumerate(lines[1:], start=2):
        parts = line.strip().split(",")
        if len(parts) < 3:
            print(f"  Skipping line {line_num}: insufficient columns")
            continue

        building_name = parts[0].strip()
        try:
            lib_lat = float(parts[1].strip())
            lib_lon = float(parts[2].strip())
        except ValueError:
            print(f"  Skipping line {line_num}: invalid coordinates")
            continue

        min_dist = float("inf")
        nearest_university = None

        for university, (uni_lat, uni_lon) in university_coords.items():
            dist = haversine_distance(lib_lat, lib_lon, uni_lat, uni_lon)
            if dist < min_dist:
                min_dist = dist
                nearest_university = university

        results.append((building_name, nearest_university))
        print(f"  {building_name} -> {nearest_university} ({min_dist:.1f} km)")

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("Building,University\n")
        for building, university in results:
            f.write(f"{building},{university}\n")

    print(f"\nDone! Results saved to {output_file}")

from geopy.distance import distance as geo_distance

def collect_spread_university_targets(
    input_file: str = "/data/feihong/ckpt/matched_university_libraries.txt",
    output_file: str = "runs/university_targets.txt",
    per_university: int = 6,
    search_radius_m: int = 2000,
    min_name_len: int = 3,
    min_sep_m: int = 300,
    sleep_between: int = 30,
) -> None:
    """
    Build a POI list where each university contributes up to `per_university`
    well-spaced buildings (>= `min_sep_m`). Results saved as CSV:
        Location,University,Latitude,Longitude
    """
    geolocator = Nominatim(user_agent="spread_university_locator", timeout=15)
    overpass_endpoints = [
        "https://overpass-api.de/api/interpreter",
        "https://overpass.kumi.systems/api/interpreter",
        "https://overpass.openrailwaymap.de/api/interpreter",
    ]

    def load_universities() -> List[str]:
        seen, universities = set(), []
        with open(input_file, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if idx == 0:
                    continue
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 2:
                    continue
                name = parts[1]
                if name and name not in seen:
                    universities.append(name)
                    seen.add(name)
        return universities

    def geocode(name: str) -> Tuple[float, float]:
        for attempt in range(3):
            try:
                loc = geolocator.geocode(name)
                if loc:
                    return loc.latitude, loc.longitude
            except (GeocoderTimedOut, GeocoderUnavailable):
                time.sleep(2 + attempt)
        raise RuntimeError(f"Geocoding failed: {name}")

    def query_overpass(lat: float, lon: float, keyword: str, endpoint_idx: int) -> List[Dict]:
        query = f"""
        [out:json][timeout:50];
        (
            nwr["building"](around:{search_radius_m},{lat},{lon});
            nwr["amenity"](around:{search_radius_m},{lat},{lon});
        );
        out center tags;
        """
        url = overpass_endpoints[endpoint_idx % len(overpass_endpoints)]
        response = requests.post(url, data={"data": query})
        response.raise_for_status()
        items, keyword_lower = [], keyword.lower()
        for el in response.json().get("elements", []):
            tags = el.get("tags", {})
            name = tags.get("name") or tags.get("name:en")
            if not name or len(name.strip()) < min_name_len:
                continue
            center = el.get("center") or el
            el_lat, el_lon = center.get("lat"), center.get("lon")
            if el_lat is None or el_lon is None:
                continue
            dist = math.hypot(el_lat - lat, el_lon - lon)
            bias = -1.0 if keyword_lower in name.lower() else 0.0
            items.append((dist + bias, {"name": name.strip(), "lat": el_lat, "lon": el_lon}))
        items.sort(key=lambda x: x[0])
        return [item for _, item in items]

    def pick_spread_points(candidates: List[Dict]) -> List[Dict]:
        picked = []
        for candidate in candidates:
            point = (candidate["lat"], candidate["lon"])
            if all(geo_distance(point, (p["lat"], p["lon"])).meters >= min_sep_m for p in picked):
                picked.append(candidate)
                if len(picked) == per_university:
                    break
        return picked

    universities = load_universities()
    rows = []
    endpoint_idx = 0

    print(f"Processing {len(universities)} universities...")
    for idx, university in enumerate(universities, start=1):
        print(f"[{idx}/{len(universities)}] {university}")
        try:
            campus_lat, campus_lon = geocode(university)
        except RuntimeError as err:
            print(f"  ! {err}")
            continue

        try:
            candidates = query_overpass(campus_lat, campus_lon, university.split()[0], endpoint_idx)
        except Exception as err:
            print(f"  ! Overpass error ({err}); rotating endpoint and retrying later.")
            endpoint_idx += 1
            time.sleep(sleep_between)
            candidates = [{
                "name": f"{university} Center",
                "lat": campus_lat,
                "lon": campus_lon,
            }]

        spread = pick_spread_points(candidates)
        if not spread:
            print(f"  - No sufficiently spaced POIs within {search_radius_m} m.")
            continue

        for item in spread:
            rows.append(f"{item['name']},{university},{item['lat']},{item['lon']}")

        endpoint_idx += 1
        time.sleep(sleep_between)

    with open(output_file, "w", encoding="utf-8") as out_f:
        out_f.write("Location,University,Latitude,Longitude\n")
        out_f.write("\n".join(rows))
    print(f"\nDone! Stored {len(rows)} entries in {output_file}")


def rewrite_targets_without_centers(
    input_file: str = "runs/university_targets.txt",
    output_file: str = "runs/university_targets_refined.txt",
    per_university: int = 6,
    search_radius_m: int = 2000,
    min_name_len: int = 3,
    min_sep_m: int = 300,
    sleep_between: int = 0,
) -> None:
    """Rewrite the targets file, refreshing rows whose location names still contain 'Center'."""

    def parse_target_line(raw_line: str) -> Tuple[str, str, str, str]:
        parts = [p.strip() for p in raw_line.split(",")]
        if len(parts) < 4:
            raise ValueError(f"Malformed line: {raw_line}")
        lon = parts[-1]
        lat = parts[-2]
        university = parts[-3]
        location = ",".join(parts[:-3]).strip()
        return location, university, lat, lon

    def load_target_file(path: str) -> Dict[str, List[str]]:
        replacements: Dict[str, List[str]] = {}
        with open(path, "r", encoding="utf-8") as src:
            for idx, line in enumerate(src):
                if idx == 0:
                    continue
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    _, university, _, _ = parse_target_line(stripped)
                except ValueError as err:
                    print(f"[WARN] {err}")
                    continue
                replacements.setdefault(university, []).append(stripped)
        return replacements

    header_line = "Location,University,Latitude,Longitude"
    records: List[Dict[str, str]] = []
    fallback_lines: Dict[str, List[str]] = {}
    centers_to_fix: List[str] = []

    with open(input_file, "r", encoding="utf-8") as src:
        for idx, raw_line in enumerate(src):
            stripped = raw_line.strip()
            if idx == 0:
                if stripped:
                    header_line = stripped
                continue
            if not stripped:
                continue

            try:
                location, university, _, _ = parse_target_line(stripped)
            except ValueError as err:
                print(f"[WARN] {err}")
                continue

            if "center" in location.lower():
                records.append({"type": "center", "university": university})
                fallback_lines.setdefault(university, []).append(stripped)
                if university not in centers_to_fix:
                    centers_to_fix.append(university)
            else:
                records.append({"type": "line", "line": stripped})

    if not centers_to_fix:
        shutil.copyfile(input_file, output_file)
        print("No 'Center' rows detected; copied file unchanged.")
        return

    tmp_input = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt")
    tmp_output = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt")
    tmp_input_path, tmp_output_path = tmp_input.name, tmp_output.name
    tmp_input.close()
    tmp_output.close()
    replacements: Dict[str, List[str]] = {}

    try:
        with open(tmp_input_path, "w", encoding="utf-8") as temp_in:
            temp_in.write("Building,University\n")
            for university in centers_to_fix:
                temp_in.write(f"{university} Refresh,{university}\n")

        collect_spread_university_targets(
            input_file=tmp_input_path,
            output_file=tmp_output_path,
            per_university=per_university,
            search_radius_m=search_radius_m,
            min_name_len=min_name_len,
            min_sep_m=min_sep_m,
            sleep_between=sleep_between,
        )

        replacements = load_target_file(tmp_output_path)
    finally:
        try:
            os.remove(tmp_input_path)
        except OSError:
            pass
        try:
            os.remove(tmp_output_path)
        except OSError:
            pass

    output_rows: List[str] = []
    replacement_consumed: Dict[str, bool] = {}
    for record in records:
        if record["type"] == "line":
            output_rows.append(record["line"])
            continue

        university = record["university"]
        if university in replacements and not replacement_consumed.get(university):
            output_rows.extend(replacements[university])
            replacement_consumed[university] = True
        else:
            fallback = fallback_lines.get(university)
            if fallback:
                output_rows.append(fallback.pop(0))

    with open(output_file, "w", encoding="utf-8") as dst:
        dst.write(header_line + "\n")
        if output_rows:
            dst.write("\n".join(output_rows))

    refreshed = sum(len(lines) for lines in replacements.values()) if centers_to_fix else 0
    print(f"Rewrote {len(output_rows)} rows to {output_file} (refreshed {refreshed} entries).")
# get_lib_coord()
# get_world_map_two()
# match_libraries()
# collect_spread_university_targets()
rewrite_targets_without_centers(
    input_file="runs/university_targets_clean_twice.txt",
    output_file="runs/university_targets_clean_third.txt",
    sleep_between=10  # optional: reduce delay for small batches
)