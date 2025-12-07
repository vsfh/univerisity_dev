import xml.etree.ElementTree as ET
from xml.dom import minidom
import os
from pathlib import Path

def extract_coordinates_from_kml_tours(directory_path, output_file="coordinates.txt"):
    """
    Extract longitude, latitude, and altitude from multiple KML tour files in a directory.
    
    Args:
        directory_path (str): Path to the directory containing KML files
        output_file (str): Output text file name
    """
    
    # Supported file extensions
    kml_extensions = {'.kml'}
    
    # Get all KML files in the directory
    kml_files = []
    for file in os.listdir(directory_path):
        if Path(file).suffix.lower() in kml_extensions:
            kml_files.append(file)
    
    if not kml_files:
        print(f"No KML files found in directory: {directory_path}")
        return
    
    print(f"Found {len(kml_files)} KML files")
    
    # Results storage
    coordinates_data = []
    
    for kml_file in kml_files:
        file_path = os.path.join(directory_path, kml_file)
        
        try:
            # Parse the KML file
            tree = ET.parse(file_path)
            root = tree.getroot()
            
            # Define namespaces
            namespaces = {
                'kml': 'http://www.opengis.net/kml/2.2',
                'gx': 'http://www.google.com/kml/ext/2.2'
            }
            
            # Find the first LookAt element (contains the coordinates)
            # Try different XPath patterns to find the LookAt element
            lookat = None
            
            # Pattern 1: Direct path to LookAt
            lookat = root.find('.//kml:LookAt', namespaces)
            if lookat is None:
                # Pattern 2: Look for LookAt in gx:FlyTo elements
                lookat = root.find('.//gx:FlyTo/kml:LookAt', namespaces)
            if lookat is None:
                # Pattern 3: More generic search
                lookat = root.find('.//{http://www.opengis.net/kml/2.2}LookAt')
            
            if lookat is not None:
                # Extract values with error handling
                longitude_elem = lookat.find('kml:longitude', namespaces)
                latitude_elem = lookat.find('kml:latitude', namespaces)
                altitude_elem = lookat.find('kml:altitude', namespaces)
                
                if longitude_elem is not None and latitude_elem is not None:
                    longitude = longitude_elem.text
                    latitude = latitude_elem.text
                    altitude = altitude_elem.text if altitude_elem is not None else "N/A"
                    
                    coordinates_data.append({
                        'filename': kml_file.split('_')[1].split('.')[0],
                        'longitude': longitude,
                        'latitude': latitude,
                        'altitude': altitude
                    })
                    print(f"✓ Extracted from {kml_file}")
                else:
                    print(f"✗ Missing coordinates in {kml_file}")
            else:
                print(f"✗ No LookAt element found in {kml_file}")
                
        except ET.ParseError as e:
            print(f"✗ Error parsing {kml_file}: {e}")
        except Exception as e:
            print(f"✗ Error processing {kml_file}: {e}")
    
    # Write results to output file
    if coordinates_data:
        with open(output_file, 'w', encoding='utf-8') as f:
            # Write header
            f.write("Filename,Longitude,Latitude,Altitude\n")
            
            # Write data
            for data in coordinates_data:
                f.write(f"{data['filename']},{data['longitude']},{data['latitude']},{data['altitude']}\n")
                name = data['filename']
                generate_custom_kml_tour(data['longitude'], data['latitude'], data['altitude'], 512, tilt=45, output_file=f'{name}_range_512.kml')
                generate_custom_kml_tour(data['longitude'], data['latitude'], data['altitude'], 256, tilt=45, output_file=f'{name}_height_180.kml')
                generate_custom_kml_tour(data['longitude'], data['latitude'], data['altitude'], 308, tilt=35, output_file=f'{name}_height_250.kml')
                generate_custom_kml_tour(data['longitude'], data['latitude'], data['altitude'], 206, tilt=54, output_file=f'{name}_height_100.kml')
        
        # print(f"\n✅ Successfully extracted coordinates from {len(coordinates_data)} files")
        print(f"📁 Output saved to: {output_file}")
        
        # Print summary
        print("\n📊 Summary:")
        for data in coordinates_data:
            print(f"  {data['filename']} {data['longitude']} {data['latitude']} {data['altitude']}")
    else:
        print("❌ No coordinates extracted from any files")

def generate_custom_kml_tour(longitude, latitude, altitude, range_val, tilt=60, fov=75, 
                           heading_increment=4, duration=0.1, output_file="custom_tour.kml"):
    """
    Generate a customizable KML tour file.
    
    Args:
        longitude (float): Longitude of the target point
        latitude (float): Latitude of the target point  
        altitude (float): Camera altitude
        range_val (float): Fixed distance from camera to target point
        tilt (float): Camera tilt angle (0-90)
        fov (float): Field of view in degrees
        heading_increment (float): Degrees to rotate each frame
        duration (float): Duration for each transition in seconds
        output_file (str): Output KML filename
    """
    
    kml = ET.Element('kml')
    kml.set('xmlns', 'http://www.opengis.net/kml/2.2')
    kml.set('xmlns:gx', 'http://www.google.com/kml/ext/2.2')
    kml.set('xmlns:kml', 'http://www.opengis.net/kml/2.2')
    kml.set('xmlns:atom', 'http://www.w3.org/2005/Atom')
    
    tour = ET.SubElement(kml, 'gx:Tour')
    name = ET.SubElement(tour, 'name')
    name.text = 'CustomRotationTour'
    
    playlist = ET.SubElement(tour, 'gx:Playlist')
    
    # Initial FlyTo
    initial_flyto = ET.SubElement(playlist, 'gx:FlyTo')
    initial_lookat = ET.SubElement(initial_flyto, 'LookAt')
    
    ET.SubElement(initial_lookat, 'gx:horizFov').text = str(fov)
    ET.SubElement(initial_lookat, 'longitude').text = str(longitude)
    ET.SubElement(initial_lookat, 'latitude').text = str(latitude)
    ET.SubElement(initial_lookat, 'altitude').text = str(altitude)
    ET.SubElement(initial_lookat, 'heading').text = "0"
    ET.SubElement(initial_lookat, 'tilt').text = str(tilt)
    ET.SubElement(initial_lookat, 'range').text = str(range_val)
    ET.SubElement(initial_lookat, 'gx:altitudeMode').text = 'relativeToSeaFloor'
    
    # Generate rotation frames
    frame_count = 0
    heading = 0
    while heading < 360:
        flyto = ET.SubElement(playlist, 'gx:FlyTo')
        
        ET.SubElement(flyto, 'gx:duration').text = str(duration)
        ET.SubElement(flyto, 'gx:flyToMode').text = 'smooth'
        
        lookat = ET.SubElement(flyto, 'LookAt')
        
        ET.SubElement(lookat, 'gx:horizFov').text = str(fov)
        ET.SubElement(lookat, 'longitude').text = str(longitude)
        ET.SubElement(lookat, 'latitude').text = str(latitude)
        ET.SubElement(lookat, 'altitude').text = str(altitude)
        ET.SubElement(lookat, 'heading').text = str(heading)
        ET.SubElement(lookat, 'tilt').text = str(tilt)
        ET.SubElement(lookat, 'range').text = str(range_val)
        ET.SubElement(lookat, 'gx:altitudeMode').text = 'relativeToSeaFloor'
        
        heading += heading_increment
        frame_count += 1
    
    # Convert to pretty XML and save
    rough_string = ET.tostring(kml, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    pretty_xml = reparsed.toprettyxml(indent="  ")
    
    with open(f'../ckpt/tour/{output_file}', 'w', encoding='utf-8') as f:
        f.write(pretty_xml)
    
    # print(f"Custom KML tour generated: {output_file}")
    # print(f"Frames: {frame_count}, Tilt: {tilt}°, FOV: {fov}°, Range: {range_val}m")
    # print(f"Rotation: 0° to {heading-heading_increment}° in {heading_increment}° increments")


# Example usage
if __name__ == "__main__":
    extract_coordinates_from_kml_tours('/home/SATA4T/gregory/data/university1652-tour')
    # generate_custom_kml_tour(
    #     longitude=-113.4689682002154,
    #     latitude=53.5214914000673,
    #     altitude=50.0,
    #     range_val=200.0,
    #     tilt=60,
    #     fov=75,
    #     heading_increment=4,
    #     duration=0.1,
    #     output_file="my_custom_tour.kml"
    # )