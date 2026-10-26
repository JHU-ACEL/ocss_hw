#!/usr/bin/env python3
"""Convert fixed URDF to MuJoCo MJCF with proper scene wrapping."""
import mujoco
import xml.etree.ElementTree as ET

import trimesh
import os
from pathlib import Path


"""

    From: modeling-and-identification-of-hover-flight-dynamics-for-nasa-s-mars-helicopter

    scaled by 1.8 for now cause that was the mass value used in 

    in-flight-system-identification-of-the-ingenuity-mars-helicopter

    Mass: 0.765 kg
    Ixx: 0.000285 kg⋅m²
    Iyy: 0.000289 kg⋅m²
    Izz: 0.000121 kg⋅m²

    mb just keep og values with an on earth scene, idk

"""





MAIN_BODY_MASS = 1.8
MAIN_BODY_IXX = 0.00067
MAIN_BODY_IYY = 0.00068
MAIN_BODY_IZZ = 0.00028

NOM_MASS = 1e-6
NOM_INERTIA = 1e-6

def get_link_inertial_vals(link_name):


    if link_name == "Body_MHS_MainBody_v16":
        return MAIN_BODY_MASS, MAIN_BODY_IXX, MAIN_BODY_IYY, MAIN_BODY_IZZ
    else:
        return NOM_MASS, NOM_INERTIA, NOM_INERTIA, NOM_INERTIA


def convert_gltf_to_obj(gltf_path, output_path=None):
    """
    Convert GLTF/GLB file to OBJ format.
    
    Args:
        gltf_path: Path to input GLTF/GLB file
        output_path: Path to output OBJ file (optional)
    """
    if output_path is None:
        output_path = gltf_path.replace('.gltf', '.obj').replace('.glb', '.obj')
    
    try:
        # Load mesh
        mesh = trimesh.load(gltf_path, force='mesh')
        
        # If it's a scene with multiple meshes, merge them
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(
                [geom for geom in mesh.geometry.values()]
            )
        
        # Export as OBJ
        mesh.export(output_path)
        print(f"✓ Converted: {os.path.basename(gltf_path)} -> {os.path.basename(output_path)}")
        return True
        
    except Exception as e:
        print(f"✗ Failed to convert {gltf_path}: {e}")
        return False

def convert_directory(input_dir, output_dir=None):
    """Convert all GLTF/GLB files in a directory."""
    if output_dir is None:
        output_dir = input_dir
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all GLTF/GLB files
    gltf_files = list(Path(input_dir).glob('*.gltf')) + list(Path(input_dir).glob('*.glb'))
    
    print(f"Found {len(gltf_files)} mesh files to convert\n")
    
    success = 0
    for gltf_file in gltf_files:
        output_file = os.path.join(output_dir, gltf_file.stem + '.obj')
        if convert_gltf_to_obj(str(gltf_file), output_file):
            success += 1
    
    print(f"\n✓ Successfully converted {success}/{len(gltf_files)} files")


def fix_urdf_mesh_references(urdf_path, output_path=None):
    """
    Replace .gltf/.glb references with .obj in URDF.
    
    Args:
        urdf_path: Path to input URDF file
        output_path: Path to output URDF file (optional, overwrites input if None)
    """
    if output_path is None:
        output_path = urdf_path
    
    with open(urdf_path, 'r') as f:
        content = f.read()
    
    # Count replacements
    gltf_count = content.count('.gltf')
    glb_count = content.count('.glb')
    
    # Replace mesh references
    content = content.replace('.gltf', '.obj')
    content = content.replace('.glb', '.obj')
    
    # Write output
    with open(output_path, 'w') as f:
        f.write(content)
    
    print(f"✓ Fixed {gltf_count + glb_count} mesh references in {urdf_path}")
    if output_path != urdf_path:
        print(f"  Saved to: {output_path}")



def fix_urdf(urdf_path, fixed_urdf_path):
    """Fix initial problems with urdf"""


    tree = ET.parse(urdf_path)
    root = tree.getroot()

    # Update mass and inertia of all links to be non-zero and reasonable
    for link in root.findall('link'):
        link_name = link.get('name')
        inertial = link.find('inertial')

        # Skip root
        if link_name == "root":
            continue

        # Add inertial if missing
        if inertial is None:
            inertial = ET.SubElement(link, 'inertial')
            ET.SubElement(inertial, 'origin', xyz="0 0 0", rpy="0 0 0")
            ET.SubElement(inertial, 'mass', value="0")  # default mass
            ET.SubElement(inertial, 'inertia', 
                          ixx="0", ixy="0", ixz="0",
                          iyy="0", iyz="0",
                          izz="0")
            print(f"Added inertial to link: {link_name}")

        mass_elem = inertial.find('mass')
        inertia_elem = inertial.find('inertia')

        mass, ixx, iyy, izz = get_link_inertial_vals(link_name)

        if mass_elem is not None:
            mass_elem.set("value", str(mass))
        
        if inertia_elem is not None:
            inertia_elem.set("ixx", str(ixx))
            inertia_elem.set("iyy", str(iyy))
            inertia_elem.set("izz", str(izz))
            # Keep off-diagonals at 0
            inertia_elem.set("ixy", "0")
            inertia_elem.set("ixz", "0")
            inertia_elem.set("iyz", "0")

    # Change MHS main body v16 joint from fixed to floating
    for joint in root.findall('joint'):

        joint_name = joint.get('name')

        if joint_name == "MHS_MainBody_v16":
            joint.set("type", "floating")
            print(f"Changed joint {joint_name} to floating")

        # Change MHS navcam to fixed joint
        elif joint_name == "MHS_NAVCAM":
            joint.set("type", "fixed")
            print(f"Changed joint {joint_name} to fixed")

    # Write fixed URDF
    tree.write(fixed_urdf_path, encoding='utf-8', xml_declaration=True)
    print(f"✓ Fixed URDF written to: {fixed_urdf_path}")



def add_collision_from_visual(urdf_path, output_path):
    """Copy visual geometries to collision tags so MuJoCo imports them."""
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    
    for link in root.findall('.//link'):
        # If link has visual but no collision, duplicate visual as collision
        visual = link.find('visual')
        collision = link.find('collision')
        
        if visual is not None and collision is None:
            collision = ET.SubElement(link, 'collision')
            # Copy origin if it exists
            visual_origin = visual.find('origin')
            if visual_origin is not None:
                collision_origin = ET.SubElement(collision, 'origin')
                for attr, val in visual_origin.attrib.items():
                    collision_origin.set(attr, val)
            # Copy geometry
            visual_geom = visual.find('geometry')
            if visual_geom is not None:
                ET.SubElement(collision, 'geometry').append(visual_geom[0])
    
    tree.write(output_path, encoding='utf-8', xml_declaration=True)

def convert_to_mjcf(urdf_path, mjcf_path):
    """Convert URDF to MJCF."""
    model = mujoco.MjModel.from_xml_path(urdf_path)
    mujoco.mj_saveLastXML(mjcf_path, model)
    print(f"✓ Converted to MJCF: {mjcf_path}")

def create_scene(mjcf_path, output_path):
    """Extract bodies from converted MJCF and wrap in Mars scene."""
    
    # Parse the converted MJCF
    tree = ET.parse(mjcf_path)
    converted_root = tree.getroot()
    converted_worldbody = converted_root.find('worldbody')
    converted_asset = converted_root.find('asset')
    
    # Create new scene
    scene = ET.Element('mujoco', model="magneto_mars")
    
    # Compiler (use degrees for readability, MuJoCo will handle conversion)
    ET.SubElement(scene, 'compiler', angle="degree", meshdir=".")
    
    # Mars physics
    ET.SubElement(scene, 'option', gravity="0 0 -3.71", timestep="0.002")
    
    # Visual settings
    visual = ET.SubElement(scene, 'visual')
    ET.SubElement(visual, 'headlight', ambient="0.5 0.5 0.5")
    
    # Assets (floor material)
    asset = ET.SubElement(scene, 'asset')
    ET.SubElement(asset, 'texture', 
                  name="grid", type="2d", builtin="checker",
                  width="512", height="512",
                  rgb1="0.8 0.6 0.4", rgb2="0.6 0.4 0.2")
    ET.SubElement(asset, 'material',
                  name="grid", texture="grid", 
                  texrepeat="10 10", reflectance="0.2")
    
    # Worldbody with floor + helicopter
    worldbody = ET.SubElement(scene, 'worldbody')
    ET.SubElement(worldbody, 'light', 
                  pos="0 0 5", dir="0 0 -1", 
                  diffuse="0.8 0.8 0.8")
    ET.SubElement(worldbody, 'geom',
                  name="floor", type="plane", 
                  size="50 50 0.1", material="grid")

    # Copy every mesh (and any material/texture) asset the compile step
    # actually produced -- this is the step the original script skipped.
    n_meshes_copied = 0
    if converted_asset is not None:
        for child in converted_asset:
            asset.append(child)
            if child.tag == 'mesh':
                n_meshes_copied += 1
    print(f"Copied {n_meshes_copied} mesh asset(s) into scene <asset> block")


    
    
    # Copy all bodies from converted file
    if converted_worldbody is not None:
        for body in converted_worldbody:
            worldbody.append(body)
    
    # Write the merged scene
    scene_tree = ET.ElementTree(scene)
    ET.indent(scene_tree, space="  ")
    scene_tree.write(output_path, encoding='utf-8', xml_declaration=True)
    print(f"✓ Created scene: {output_path}")



def fix_scene(input_path: str, output_path: str) -> None:
    """
    Convert helicopter XML model to simplified version.
    
    Args:
        input_path: Path to original scene.xml
        output_path: Path for output simplified XML
    """
    # Parse XML
    tree = ET.parse(input_path)
    root = tree.getroot()
    
    # Body names
    MAIN_BODY = "Body_MHS_MainBody_v16"
    BLADE_BODIES = ["Body_MHS_TopBlades_v16", "Body_MHS_BottomBlades_v16"]
    LEG_BODIES = [
        "Body_Joint_Leg02_Axis",
        "Body_Joint_Leg01_Axis", 
        "Body_Joint_Leg03_Axis",
        "Body_Joint_Leg04_Axis"
    ]
    CHILD_BODIES = BLADE_BODIES + LEG_BODIES
    
    # # Corrected values
    # BLADE_MASS = "0.05"
    # LEG_MASS = "0.02"
    # BLADE_DIAGINERTIA = "0.0001 0.0001 0.0001"
    
    # Find all body elements
    for body in root.iter("body"):
        body_name = body.get("name")
        
        # 1. Add pos to main body
        if body_name == MAIN_BODY:
            if body.get("pos") is None:
                body.set("pos", "0 0 1")
                print(f"Added pos='0 0 1' to {body_name}")
        
        # 2. Remove joints from child bodies
        if body_name in CHILD_BODIES:
            joints_to_remove = list(body.findall("joint"))
            for joint in joints_to_remove:
                body.remove(joint)
                print(f"Removed joint from {body_name}")
        
        # # 3. Fix masses and inertias
        # if body_name in BLADE_BODIES:
        #     for inertial in body.findall("inertial"):
        #         inertial.set("mass", BLADE_MASS)
        #         inertial.set("diaginertia", BLADE_DIAGINERTIA)
        #         print(f"Fixed mass={BLADE_MASS}, diaginertia for {body_name}")
        
        # elif body_name in LEG_BODIES:
        #     for inertial in body.findall("inertial"):
        #         inertial.set("mass", LEG_MASS)
        #         print(f"Fixed mass={LEG_MASS} for {body_name}")
    
    # Write output
    tree.write(output_path, encoding="unicode", xml_declaration=True)
    print(f"\nWrote simplified model to: {output_path}")












if __name__ == "__main__":
    # urdf_path = "models/mhs/MHS.urdf"
    # fix_1_urdf_path = "models/mhs/mhs_temp.urdf"
    # fixed_urdf_path = "models/mhs/mhs_temp_fixed.urdf"
    # mjcf_path = "models/mhs/mhs_temp.xml"
    # scene_path = "models/mhs/mhs_scene.xml"
    # scene_fixed_path = "models/mhs/mhs_scene_fixed.xml"

    meshes_init = "m2020-urdf-models/mhs/meshes/"
    meshes_final = "models/mhs/meshes/"
    
    if os.path.isdir(meshes_init):
        convert_directory(meshes_init, meshes_final)
    else:
        convert_gltf_to_obj(meshes_init, meshes_final)


    urdf_path = "m2020-urdf-models/mhs/MHS.urdf"
    urdf_temp_path = "models/mhs/mhs_temp.urdf"
    mhs_path = "models/mhs/mhs_scene.xml"

    fix_urdf_mesh_references(urdf_path, urdf_temp_path)
    fix_urdf(urdf_temp_path, urdf_temp_path)
    add_collision_from_visual(urdf_temp_path, urdf_temp_path)
    convert_to_mjcf(urdf_temp_path, mhs_path)
    create_scene(mhs_path, mhs_path)
    fix_scene(mhs_path, mhs_path)

    # fix_urdf(urdf_path, fix_1_urdf_path)
    # add_collision_from_visual(fix_1_urdf_path, fixed_urdf_path)
    # convert_to_mjcf(fixed_urdf_path, mjcf_path)
    # create_scene(mjcf_path, scene_path)
    # fix_scene(scene_path, scene_fixed_path)
