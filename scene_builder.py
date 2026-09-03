"""
scene_builder.py
================
Run once after each FreeCAD export to compile the raw assembly.xml
into a physics-ready scene.xml. Does NOT modify FreeCAD source files.

Usage:
python scene_builder.py
"""

import xml.etree.ElementTree as ET
import numpy as np
import struct
import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ASSEMBLY_XML = "3D_files/mujoco/assembly.xml"
CAMERAS_XML = "3D_files/mujoco/cameras.xml"
MESHES_DIR = "3D_files/mujoco/meshes"
OUTPUT_XML = "3D_files/mujoco/scene.xml"

# ---------------------------------------------------------------------------
# Physics constants
# ---------------------------------------------------------------------------
MAX_WHEEL_VEL = 29.24  # rad/s  ← 10 km/h with 190mm diameter wheels
MAX_STEER_ANG = 0.7854  # rad    ← ±45 degrees
WHEEL_MASS_KG = 0.3  # per wheel (rubber + hub estimate)
CHASSIS_AREA_KG_PER_M2 = (
    5.4  # 2mm aluminum hollow shell: 2.7 g/cm³ × 0.2cm = 0.54 g/cm² = 5.4 kg/m²
)

# Joints we monitor as passive sensors (4 rocker-bogie angles)
PASSIVE_SENSOR_JOINTS = [
    "pass_left_rocker",
    "pass_right_rocker",
    "pass_left_rockerbogie",
    "pass_right_rockerbogie",
]


# ---------------------------------------------------------------------------
# Mesh surface area estimator (OBJ files only)
# ---------------------------------------------------------------------------
def obj_surface_area(filepath: str) -> float:
    """
    Parses a .obj file and returns the total surface area in m².
    Assumes the .obj is exported in millimetres (FreeCAD default) and converts.
    """
    verts = []
    area = 0.0
    with open(filepath, "r") as f:
        for line in f:
            if line.startswith("v "):
                _, x, y, z = line.split()
                verts.append(
                    np.array([float(x), float(y), float(z)]) / 1000.0
                )  # mm → m
            elif line.startswith("f "):
                idx = [int(p.split("/")[0]) - 1 for p in line.split()[1:]]
                for i in range(1, len(idx) - 1):
                    a = verts[idx[0]]
                    b = verts[idx[i]]
                    c = verts[idx[i + 1]]
                    area += 0.5 * np.linalg.norm(np.cross(b - a, c - a))
    return area


# ---------------------------------------------------------------------------
# Build the final scene
# ---------------------------------------------------------------------------
print("=== scene_builder.py ===")

# 1. Load assembly
tree = ET.parse(ASSEMBLY_XML)
root = tree.getroot()

# 2. Load cameras
cam_tree = ET.parse(CAMERAS_XML)
cam_root = cam_tree.getroot()

# -----------------------------------------------------------------------
# 3. Fix collisions: remove contype="0" conaffinity="0" from ALL mesh geoms
#    so wheels / chassis can contact the terrain.
#    We'll re-add them selectively for visual-only geoms we inject later.
# -----------------------------------------------------------------------
for geom in root.iter("geom"):
    name = geom.get("name", "")
    if "floor" in name or "groundplane" in name:
        continue
    geom.attrib.pop("contype", None)
    geom.attrib.pop("conaffinity", None)
print("  [✓] Collisions enabled on all mesh geoms.")

# -----------------------------------------------------------------------
# 4. Mass calculation for each part from OBJ surface area
# -----------------------------------------------------------------------
asset = root.find("asset")
for mesh_tag in asset.findall("mesh"):
    mesh_name = mesh_tag.get("name", "")
    obj_path = os.path.join(MESHES_DIR, f"{mesh_name}.obj")
    if not os.path.exists(obj_path):
        continue
    area_m2 = obj_surface_area(obj_path)
    mass_kg = area_m2 * CHASSIS_AREA_KG_PER_M2

    # Find the geom that uses this mesh and set its mass
    for geom in root.iter("geom"):
        if geom.get("mesh") == mesh_name:
            geom.set("mass", f"{mass_kg:.6f}")
            break

    # Wheels get high friction, and hard-coded to 200g (0.2kg)
    if "wheel" in mesh_name:
        for geom in root.iter("geom"):
            if geom.get("mesh") == mesh_name:
                if "rotator" not in mesh_name:
                    geom.set("mass", "0.2")  # Real wheels hardcoded to 200g
                
                geom.set("friction", "2.0 0.005 0.0001")  # Stop Tokyo drifts!
                break

print("  [✓] Mass assigned from OBJ surface areas (2mm Al shell).")

# -----------------------------------------------------------------------
# 5. Rebuild actuator block
# -----------------------------------------------------------------------
drv_joints = []
srv_joints = []
pass_joints = []
for joint in root.iter("joint"):
    name = joint.get("name", "")
    if name.startswith("drv_"):
        drv_joints.append(name)
    elif name.startswith("srv_"):
        srv_joints.append(name)
    elif name.startswith("pass_"):
        pass_joints.append(name)

actuator = root.find("actuator")
if actuator is None:
    actuator = ET.SubElement(root, "actuator")
else:
    actuator.clear()

# Drive wheels: velocity-controlled
for jname in drv_joints:
    ET.SubElement(
        actuator,
        "velocity",
        {
            "name": jname,
            "joint": jname,
            "kv": "5",
            "forcelimited": "true",
            "forcerange": "-10 10",
            "gear": "-1",
        },
    )

# Steering: position-controlled
for jname in srv_joints:
    ET.SubElement(
        actuator,
        "position",
        {
            "name": jname,
            "joint": jname,
            "kp": "500",
            "gear": str(-MAX_STEER_ANG),  # Flipped steering direction
        },
    )

print(
    f"  [✓] Motors rebuilt: {len(drv_joints)} drive, {len(srv_joints)} steering, "
    f"{len(pass_joints)} passive (no actuator)."
)

# -----------------------------------------------------------------------
# 6. Add joint damping AND STIFFNESS to all passive suspension joints 
#    so they don't fold up and lift the middle wheel.
# -----------------------------------------------------------------------
for joint in root.iter("joint"):
    name = joint.get("name", "")
    if name.startswith("pass_"):
        joint.set("damping", "20.0")   # High damping
        joint.set("stiffness", "100.0") # High stiffness so it holds its shape
        joint.set("armature", "0.01")
    elif name.startswith("drv_"):
        joint.set("damping", "0.5")
        joint.set("armature", "0.01")
    elif name.startswith("srv_"):
        joint.set("damping", "2.0")
        joint.set("armature", "0.01")
print("  [✓] Joint damping tuned.")

# -----------------------------------------------------------------------
# 7. Sensors: IMU on chassis + jointpos for the 4 passive suspension arms
# -----------------------------------------------------------------------
sensor = root.find("sensor")
if sensor is None:
    sensor = ET.SubElement(root, "sensor")
else:
    sensor.clear()

# IMU: framequat + frameangvel on the chassis body
ET.SubElement(
    sensor, "framequat", {"name": "imu_quat", "objtype": "body", "objname": "body"}
)
ET.SubElement(
    sensor, "frameangvel", {"name": "imu_angvel", "objtype": "body", "objname": "body"}
)

# Passive joint angle sensors (virtual rotary encoders)
for jname in PASSIVE_SENSOR_JOINTS:
    ET.SubElement(sensor, "jointpos", {"name": f"sensor_{jname}", "joint": jname})

print(f"  [✓] Sensors: IMU (quat + angvel) + {len(PASSIVE_SENSOR_JOINTS)} jointpos.")

# -----------------------------------------------------------------------
# 8. Inject cameras + freejoint into the chassis body, and add Target Marker
# -----------------------------------------------------------------------
chassis = root.find(".//body[@name='body']")
if chassis is not None:
    if chassis.find("freejoint") is None:
        chassis.insert(0, ET.Element("freejoint", {"name": "root"}))
    chassis.set("pos", "0 0 0.5")
    for cam in cam_root.findall("camera"):
        chassis.append(cam)
    print(
        f"  [✓] Injected freejoint + {len(cam_root.findall('camera'))} cameras into chassis body."
    )
else:
    print("  [!] WARNING: Could not find body named 'body'. Cameras NOT injected.")

# Add target marker to worldbody (Yellow Sphere)
worldbody = root.find("worldbody")
if worldbody is not None:
    ET.SubElement(worldbody, "site", {
        "name": "target_marker",
        "type": "sphere",
        "size": "0.5",
        "pos": "0 0 0.299",  # Floor is at Z=-0.201. Center = 0.5 - 0.201 = 0.299
        "rgba": "1 1 0 0.5"  # Yellow, semi-transparent
    })
    print("  [✓] Injected yellow target marker sphere.")

# -----------------------------------------------------------------------
# 9. Write scene.xml
# -----------------------------------------------------------------------
ET.indent(root, space="  ")
tree.write(OUTPUT_XML, encoding="unicode", xml_declaration=True)
print(f"  [✓] Written to {OUTPUT_XML}")
print("=== Done! ===")
