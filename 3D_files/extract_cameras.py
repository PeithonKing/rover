import zipfile
import xml.etree.ElementTree as ET
import os

FCSTD_PATH = "parts.FCStd"
OUTPUT_PATH = "mujoco/cameras.xml"

def extract_cameras():
    print(f"Extracting coordinates from {FCSTD_PATH}...")
    
    with zipfile.ZipFile(FCSTD_PATH, 'r') as z:
        with z.open('Document.xml') as f:
            tree = ET.parse(f)
            root = tree.getroot()

    object_data = root.find('ObjectData')
    
    left_pos = None
    right_pos = None

    for obj in object_data.findall('Object'):
        label_prop = obj.find(".//Property[@name='Label']/String")
        if label_prop is not None:
            label = label_prop.attrib.get('value', '')
            if label in ['left_camera', 'right_camera']:
                placement = obj.find(".//Property[@name='Placement']/PropertyPlacement")
                if placement is not None:
                    px = float(placement.attrib['Px']) / 1000.0
                    py = float(placement.attrib['Py']) / 1000.0
                    pz = float(placement.attrib['Pz']) / 1000.0
                    pos_str = f"{px} {py} {pz}"
                    
                    if label == 'left_camera':
                        left_pos = pos_str
                    elif label == 'right_camera':
                        right_pos = pos_str
                        
    if not left_pos or not right_pos:
        print("Error: Could not find left_camera or right_camera in parts.FCStd")
        return

    print(f"Found left_camera at:  {left_pos}")
    print(f"Found right_camera at: {right_pos}")

    xml_content = f"""<!-- AUTOMATICALLY GENERATED CAMERAS FROM FREECAD FCSTD -->
<mujoco>
    <!-- Front Left Camera (Points +X) -->
    <camera name="cam_front_left" pos="{left_pos}" xyaxes="0 -1 0 0 0 1" fovy="60"/>
    
    <!-- Side Left Camera (Points +Y) -->
    <camera name="cam_side_left" pos="{left_pos}" xyaxes="1 0 0 0 0 1" fovy="60"/>
    
    <!-- Front Right Camera (Points +X) -->
    <camera name="cam_front_right" pos="{right_pos}" xyaxes="0 -1 0 0 0 1" fovy="60"/>
    
    <!-- Side Right Camera (Points -Y) -->
    <camera name="cam_side_right" pos="{right_pos}" xyaxes="-1 0 0 0 0 1" fovy="60"/>
</mujoco>
"""

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, 'w') as f:
        f.write(xml_content)
        
    print(f"Successfully generated {OUTPUT_PATH} beside assembly.xml!")

if __name__ == "__main__":
    extract_cameras()
