import matplotlib.pyplot as plt
import geopandas as gpd
import os
import xml.etree.ElementTree as ET
import requests

def get_lib_coord():
    import time
    from geopy.geocoders import Nominatim
    from geopy.exc import GeocoderTimedOut

    # Initialize OpenStreetMap's free geocoder
    # Make sure to keep a unique user_agent name here
    geolocator = Nominatim(user_agent="asian_university_locator")

    # Add your top 200 list here
    asian_top_200 = [
        "Peking University",
        "The University of Hong Kong",
        "National University of Singapore",
        "Nanyang Technological University",
        "Fudan University",
        "The Chinese University of Hong Kong",
        "Tsinghua University",
        "Zhejiang University",
        "Yonsei University",
        "City University of Hong Kong",
        "The Hong Kong University of Science and Technology",
        "Universiti Malaya",
        "Korea University",
        "Shanghai Jiao Tong University",
        "Korea Advanced Institute of Science and Technology",
        "Sungkyunkwan University",
        "The Hong Kong Polytechnic University",
        "Seoul National University",
        "Hanyang University",
        "Universiti Putra Malaysia",
        "The University of Tokyo",
        "Pohang University of Science And Technology",
        "Kyoto University",
        "Nanjing University",
        "Tohoku University",
        "National Taiwan University",
        "Universiti Kebangsaan Malaysia",
        "Universiti Teknologi Malaysia",
        "Al-Farabi Kazakh National University",
        "Tokyo Institute of Technology",
        "University of Science and Technology of China",
        "Nagoya University",
        "Osaka University",
        "Kyushu University",
        "Hokkaido University",
        "Taylor's University",
        "Universiti Sains Malaysia",
        "Wuhan University",
        "National Tsing Hua University",
        "Kyung Hee University",
        "National Cheng Kung University",
        "National Yang Ming Chiao Tung University",
        "Tongji University",
        "Indian Institute of Technology Delhi",
        "UCSI University",
        "Universitas Indonesia",
        "Chulalongkorn University",
        "Indian Institute of Technology Bombay",
        "Keio University",
        "Sun Yat-sen University",
        "Waseda University",
        "Airlangga University",
        "Gadjah Mada University",
        "Universiti Teknologi PETRONAS",
        "Mahidol University",
        "Indian Institute of Technology Madras",
        "Tianjin University",
        "Harbin Institute of Technology",
        "Bandung Institute of Technology",
        "Indian Institute of Technology Kharagpur",
        "Beijing Normal University",
        "Indian Institute of Science",
        "University of Tsukuba",
        "Beijing Institute of Technology",
        "L.N. Gumilyov Eurasian National University",
        "National Taiwan University of Science and Technology",
        "Indian Institute of Technology Kanpur",
        "National University of Sciences And Technology Islamabad",
        "Huazhong University of Science and Technology",
        "Ewha Womans University",
        "Hong Kong Baptist University",
        "Xi'an Jiaotong University",
        "Chung-Ang University",
        "Sunway University",
        "Shandong University",
        "Kobe University",
        "Xiamen University",
        "Universiti Brunei Darussalam",
        "National Taiwan Normal University",
        "Universiti Utara Malaysia",
        "Pusan National University",
        "University of Delhi",
        "National Sun Yat-sen University",
        "Quaid-i-Azam University",
        "Shanghai University",
        "University of the Philippines",
        "University of Tehran",
        "Ulsan National Institute of Science and Technology",
        "Satbayev University",
        "Jilin University",
        "National Taipei University of Technology",
        "IPB University",
        "Sogang University",
        "Sichuan University",
        "Daegu Gyeongbuk Institute of Science and Technology",
        "Hiroshima University",
        "Sharif University of Technology",
        "Nankai University",
        "University of Macau",
        "Sejong University",

        "Beihang University",
        "South China University of Technology",
        "East China Normal University",
        "Dalian University of Technology",
        "Southeast University",
        "Central South University",
        "Renmin University of China",
        "Chongqing University",
        "University of Electronic Science and Technology of China",
        "Hunan University",
        "Nanjing University of Aeronautics and Astronautics",
        "Jinan University",
        "Shenzhen University",
        "Soochow University",
        "Beijing University of Technology",
        "Nanjing University of Science and Technology",
        "Northeastern University",
        "Lanzhou University",
        "China Agricultural University",
        "East China University of Science and Technology",
        "Chiba University",
        "Okayama University",
        "Nagasaki University",
        "Kumamoto University",
        "Kanazawa University",
        "Niigata University",
        "Osaka Metropolitan University",
        "Tokyo Medical and Dental University",
        "Yokohama National University",
        "Kyushu Institute of Technology",
        "Gunma University",
        "Tokyo University of Science",
        "Hankuk University of Foreign Studies",
        "Inha University",
        "Kyungpook National University",
        "Chonnam National University",
        "Jeonbuk National University",
        "Ajou University",
        "Dongguk University",
        "Gyeongsang National University",
        "Hallym University",
        "Chungnam National University",
        "National Central University",
        "National Chung Hsing University",
        "National Chung Cheng University",
        "Kaohsiung Medical University",
        "Taipei Medical University",
        "National Taiwan Ocean University",
        "Feng Chia University",
        "Indian Institute of Technology Roorkee",
        "Indian Institute of Technology Guwahati",
        "University of Hyderabad",
        "Jadavpur University",
        "Savitribai Phule Pune University",
        "Anna University",
        "Banaras Hindu University",
        "Aligarh Muslim University",
        "Jamia Millia Islamia",
        "Amrita Vishwa Vidyapeetham",
        "Manipal Academy of Higher Education",
        "Vellore Institute of Technology",
        "Universiti Malaysia Pahang",
        "Universiti Malaysia Sarawak",
        "Universiti Malaysia Sabah",
        "Management and Science University",
        "Universiti Tenaga Nasional",
        "Tunku Abdul Rahman University",
        "Diponegoro University",
        "Brawijaya University",
        "Padjadjaran University",
        "Sebelas Maret University",
        "Sepuluh Nopember Institute of Technology",
        "Hasanuddin University",
        "Telkom University",
        "Ateneo de Manila University",
        "De La Salle University",
        "University of Santo Tomas",
        "Chiang Mai University",
        "Thammasat University",
        "Kasetsart University",
        "Prince of Songkla University",
        "King Mongkut's University of Technology Thonburi",
        "Lahore University of Management Sciences",
        "University of the Punjab",
        "COMSATS University Islamabad",
        "University of Engineering & Technology Lahore",
        "Aga Khan University",
        "Bangladesh University of Engineering and Technology",
        "University of Dhaka",
        "North South University",
        "BRAC University",
        "Vietnam National University, Hanoi",
        "Vietnam National University, Ho Chi Minh City",
        "Ton Duc Thang University",
        "Duy Tan University",
        "University of Colombo",
        "University of Peradeniya",
        "Macao Polytechnic University",
        "Macau University of Science and Technology",
        "National University of Mongolia"
        ]

    south_america = ["Universidade de São Paulo",
    "Pontificia Universidad Católica de Chile",
    "Universidade Estadual de Campinas",
    "Universidade Federal do Rio de Janeiro",
    "Universidad de Chile",
    "Universidad de los Andes Colombia",
    "UNESP",
    "Universidad de Buenos Aires",
    "Universidad de Concepción",
    "Universidad Nacional de Colombia",
    "Universidade Federal de Minas Gerais",
    "Pontifícia Universidade Católica do Rio de Janeiro",
    "Pontificia Universidad Católica del Perú",
    "Universidade Federal do Rio Grande Do Sul",
    "Universidad de Santiago de Chile",
    "Pontificia Universidad Javeriana",
    "Universidad de Antioquia",
    "Universidade Federal de Santa Catarina",
    "Universidad Nacional de La Plata",
    "Universidad Adolfo Ibáñez",
    "Pontificia Universidad Católica de Valparaíso",
    "Universidade de Brasília",
    "Pontificia Universidad Católica Argentina",
    "Universidad Austral",
    "Universidad de Palermo",
    "Universidade Federal de São Paulo",
    "Universidad de Belgrano",
    "Universidad Nacional de Córdoba",
    "Universidad Nacional Mayor de San Marcos",
    "Instituto Tecnológico de Buenos Aires",
    "Universidad Nacional de Rosario",
    "Universidad Torcuato di Tella",
    "Universidad de San Andrés",
    "Universidad Técnica Federico Santa María",
    "Universidad de los Andes Chile"
    ]
    output_file = "asian_200_libraries_free.txt"
    output_file = "south_america_50_libraries_free.txt"

    def get_free_coordinates(university_name):
        """Tries to find the library first, then falls back to the main campus."""
        
        campus = geolocator.geocode(university_name, timeout=10)
        if not campus:
            return "Campus not found", None, None, "Failed at Step 1"
        
        campus_lat = campus.latitude
        campus_lon = campus.longitude
        


        # STEP 2: Use Overpass API to find the exact building polygon near that point
        # We search within an 800-meter radius (around:800) for anything tagged as a library
        overpass_url =  "https://overpass-api.de/api/interpreter"
        
        overpass_query = f"""
            [out:json][timeout:25];
            (
            nwr["amenity"="library"](around:800, {campus_lat}, {campus_lon});
            nwr["building"="library"](around:800, {campus_lat}, {campus_lon});
            nwr["name"~"Library|图书馆", i](around:800, {campus_lat}, {campus_lon});
            );
            out center; 
            """
            
        try:
            response = requests.post("https://overpass-api.de/api/interpreter", data={'data': overpass_query})
            
            if response.status_code != 200:
                return f"{university_name} Campus Center", campus_lat, campus_lon, "Fallback: Server Blocked"
                
            data = response.json()
            
            if data and 'elements' in data and len(data['elements']) > 0:
                import math
                best_match = None
                closest_dist = float('inf')
                
                # Grab the first word of the university (e.g., "Peking" or "Tsinghua") 
                # to check if it's in the library's name
                search_keyword = university_name.split()[0].lower()
                
                for element in data['elements']:
                    # Overpass puts coords in 'center' for polygons, and direct lat/lon for nodes
                    lat = element.get('center', element).get('lat')
                    lon = element.get('center', element).get('lon')
                    
                    tags = element.get('tags', {})
                    name = tags.get('name', '')
                    name_en = tags.get('name:en', '')
                    
                    # Calculate rough distance from the campus center
                    dist = math.hypot(lat - campus_lat, lon - campus_lon)
                    
                    # If the building's name explicitly contains "Peking" (or whatever the university is),
                    # we artificially drop the distance score so it is guaranteed to win.
                    if search_keyword in name.lower() or search_keyword in name_en.lower():
                        dist -= 1000 
                        
                    if dist < closest_dist:
                        closest_dist = dist
                        best_match = (name or name_en or f"{university_name} Library", lat, lon)
                        
                if best_match:
                    return best_match[0], best_match[1], best_match[2], "Exact Polygon Center"
                    
            # If the loop finds nothing, fallback
            return f"{university_name} Campus Center", campus_lat, campus_lon, "Fallback to Campus"
            
        except requests.exceptions.JSONDecodeError:
            return f"{university_name} Campus Center", campus_lat, campus_lon, "Fallback: JSON Error"
        except Exception as e:
            return "API Error", campus_lat, campus_lon, "Fallback to Campus"
    # Run the automation and save to a text file
    with open(output_file, "w", encoding="utf-8") as file:
        # Notice we added a "Search Used" column so you know if it grabbed the library or the campus
        file.write("University, Matched Location, Latitude, Longitude, Search Used\n")
        
        for school in south_america:
            print(f"Searching for: {school}...")
            res = get_free_coordinates(school)
            if res[-1] == 'Exact Polygon Center':
                match_name, lat, lon, express = res
            else:
                continue
            
            file.write(f"{match_name}, {lat}, {lon}\n")
            
            # CRITICAL: OpenStreetMap is free, but they will temporarily block you 
            # if you search too fast. A 1.5 second delay keeps you safe.
            time.sleep(1.5)

    print(f"\nSuccess! All coordinates saved to {output_file}")

def process_kml_file(input_path):
    """处理单个KML文件"""
    # 获取文件名（不带扩展名）
    filename = os.path.splitext(os.path.basename(input_path))[0]
    
    # 解析KML文件
    tree = ET.parse(input_path)
    root = tree.getroot()
    
    # 定义命名空间（不同KML文件可能使用不同的命名空间）
    ns = {'kml': 'http://www.opengis.net/kml/2.2'}
    
    # 修改name元素为文件名
    name_elem = root.find('.//kml:name', ns)
    if name_elem is not None:
        name_elem.text = filename
    
    # 处理所有Placemark的LookAt元素
    for lookat in root.findall('.//kml:LookAt', ns):
            longitude_elem = lookat.find('kml:longitude', ns)
            latitude_elem = lookat.find('kml:latitude', ns)
            
            # Extract text and convert to float
            if longitude_elem is not None and latitude_elem is not None:
                lon_float = float(longitude_elem.text)
                lat_float = float(latitude_elem.text)

    return lat_float, lon_float

def get_world_map():
    input_folder = '/data/feihong/kml_1024'
    kml_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.lower().endswith('.kml')]

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
    url = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
    world = gpd.read_file(url)

    # Create the plot figure and specify the size
    fig, ax = plt.subplots(figsize=(15, 10))

    # Draw the world map
    world.plot(ax=ax, color='lightgray', edgecolor='white')

    # Plot the coordinates as red scatter points
    # Matplotlib expects (X, Y) which translates to (Longitude, Latitude)
    ax.scatter(lons, lats, color='red', s=50, zorder=5, alpha=0.8)

    input_folder = '/data/feihong/new_univ_kml'
    kml_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.lower().endswith('.kml')]

    print(f"找到 {len(kml_files)} 个KML文件需要处理")

    # Sample data: Lists of latitudes and longitudes
    lats = []
    lons = []
    for kml_file in kml_files:
        input_path = os.path.join(input_folder, kml_file)
        lat, lon = process_kml_file(input_path)
        lats.append(lat)
        lons.append(lon)

    ax.scatter(lons, lats, color='green', s=50, zorder=5, alpha=0.8)

    # Add a title
    # plt.title("Locations of University", fontsize=16)

    # Remove the axis ticks (makes it look cleaner)
    ax.set_xticks([])
    ax.set_yticks([])

    # Save the map as a PNG image
    output_image = "static_world_map.png"
    plt.savefig(output_image, bbox_inches='tight', dpi=300)

    print(f"Map saved successfully as '{output_image}'!")

# get_lib_coord()
get_world_map()