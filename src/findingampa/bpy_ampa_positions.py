import bpy
import numpy as np

from bpy.types import (Panel, Operator)

import sys
sys.path.append(sys.argv[sys.argv.index("--findingampa") + 1])


# ------------------------------------------------------------------------
#    Operators
# ------------------------------------------------------------------------
bpy.types.Scene.FA_position_name = bpy.props.StringProperty(
    name='Name',
    description='Name of the positions.',
    default='default',
    maxlen=0
)
bpy.types.Scene.FA_membrane_name = bpy.props.StringProperty(
    name='Name',
    description='Name of the membrane.',
    default='default',
    maxlen=0
)

def ShowMessageBox(message = "", title = "Message Box", icon = 'INFO'):

        def draw(self, context):
            self.layout.label(text=message)

        bpy.context.window_manager.popup_menu(draw, title = title, icon = icon)

class ShowDistance(Operator):

    bl_idname = "scene.show_dis"
    bl_label = "Show Distance"

    def execute(self, context):

    

        mode = bpy.context.active_object.mode
        # we need to switch from Edit mode to Object mode so the selection gets updated
        bpy.ops.object.mode_set(mode='OBJECT')
        selectedVerts = [v for v in bpy.context.active_object.data.vertices if v.select]
        if len(selectedVerts) != 2:
            ShowMessageBox("Please select only 2 vertices")
        else:
            ShowMessageBox(f"The distance is {np.linalg.norm(selectedVerts[0].co-selectedVerts[1].co)*100} A")
        # back to whatever mode we were in
        bpy.ops.object.mode_set(mode='EDIT')
        return {'FINISHED'}

def save_membrane(name):
    selected_objects = bpy.context.selected_objects
    if len(selected_objects) == 0:
        return {'CANCELLED'}
    all = []
    for so in selected_objects:           
        vertices = so.data.vertices
        coords = [v.co for v in vertices]
        all.append(np.array(coords))
    combined_array = np.concatenate(all)  
    np.savetxt(f"annotated_membranes/{name}.txt", combined_array*10.0)

class SavePRE(Operator):
    bl_idname = "scene.save_pre"
    bl_label = "Save presynaptic membrane"

    def execute(self, context):

        save_membrane("presynaptic")
        
        return {'FINISHED'}

class SavePOST(Operator):
    bl_idname = "scene.save_post"
    bl_label = "Save postynaptic membrane"

    def execute(self, context):
        save_membrane("postsynaptic")
        
        return {'FINISHED'}

class SaveMEM(Operator):
    bl_idname = "scene.save_mem"
    bl_label = "Save generic membrane"

    def execute(self, context):
        save_membrane(context.scene.FA_membrane_name)
        
        return {'FINISHED'}
class SaveAMPA(Operator):
    """Print object name in Console"""
    bl_idname = "scene.save_ampa"
    bl_label = "Save AMPA positions"

    def execute(self, context):
        import molecularnodes as mn
        try:
            import starfile
        except ModuleNotFoundError:
            mn.pkg.install_package('starfile')
            import starfile
        import bpy
        import numpy as np
        import pandas as pd
        try:
            from scipy.spatial.transform import Rotation as R
        except ModuleNotFoundError:
            mn.pkg.install_package("scipy")
            from scipy.spatial.transform import Rotation as R
        from findingampa.utils.imod import get_tomopitch_angle_offset
        from subprocess import Popen, PIPE
        try:
            import imodmodel
        except ModuleNotFoundError:
            mn.pkg.install_package("imodmodel")
            import imodmodel

        ampa_positions_object = bpy.data.objects['receptor_positions']
        depsgraph = bpy.context.evaluated_depsgraph_get()
        ampa_positions_object_eval = ampa_positions_object.evaluated_get(depsgraph)

        num_ampa = len(ampa_positions_object_eval.data.vertices)
        coordinates = np.ones(num_ampa*3)
        rotations = np.ones(num_ampa*4)

        ampa_positions_object_eval.data.vertices.foreach_get("co",coordinates)
        coordinates = coordinates.reshape(-1,3)

        ampa_positions_object_eval.data.attributes["rotation"].data.foreach_get("value",rotations)
        rotations = rotations.reshape(-1,4)

        #Convert from blender to scipy conventions and then into Scipy rotation
        rotations_obj = R.from_quat(rotations[:, [1, 2, 3, 0]])

        #Invert to go from blender to relion
        rotations = rotations_obj.inv()

        #Convert to euler
        rotations = rotations.as_euler(seq='ZYZ',degrees=True)

        #Convert coordinates to A
        coordinates_A = coordinates * 100

        # Write relion star file
        results_relion = []
        for i in range(num_ampa):
            results_relion.append({
                            "rlnCoordinateX": coordinates_A[i][0] / 2.5,
                            "rlnCoordinateY": coordinates_A[i][1] / 2.5,
                            "rlnCoordinateZ": coordinates_A[i][2] / 2.5,
                            "rlnOriginXAngst": 0,
                            "rlnOriginYAngst": 0,
                            "rlnOriginZAngst": 0,
                            "rlnAngleRot": rotations[i][0],
                            "rlnAngleTilt": rotations[i][1],
                            "rlnAnglePsi": rotations[i][2],
                            "rlnImagePixelSize": 2.5,
                            "rlnMicrographName": path,
                            "rlnOpticsGroup": 1,
                        })

        starfile.write({
                    'particles': pd.DataFrame(results_relion), 
                    'optics': pd.DataFrame([
                        {'rlnOpticsGroup':1}
                        ])
                }, f"ampa_positions/{context.scene.FA_position_name}_flat.star")

        # Write 3 point .txt file
        with open(f"ampa_positions/{context.scene.FA_position_name}_flat.txt", "w") as f:
            for i in range(num_ampa):
                f.write(f"{i+1} {coordinates_A[i][0] / 10} {coordinates_A[i][1] / 10} {coordinates_A[i][2] / 10}\n")
                aunp_center_point = rotations_obj[i].apply([0,0,200])
                f.write(f"{i+1} {coordinates_A[i][0] / 10 + aunp_center_point[0] / 10} {coordinates_A[i][1] / 10 + aunp_center_point[1] / 10} {coordinates_A[i][2] / 10 + aunp_center_point[2] / 10}\n")
                aunp_point = rotations_obj[i].apply([60,0,200])
                f.write(f"{i+1} {coordinates_A[i][0] / 10 + aunp_point[0] / 10} {coordinates_A[i][1] / 10 + aunp_point[1] / 10} {coordinates_A[i][2] / 10 + aunp_point[2] / 10}\n")
        pro1 = Popen(f"point2model -im {path} ampa_positions/{context.scene.FA_position_name}_flat.txt ampa_positions/{context.scene.FA_position_name}_flat.mod", stdout=PIPE, stderr=PIPE, shell=True)
        pro1.communicate()
        # Transform the 3 point model to untilted coordinates
        tilt_angle = get_tomopitch_angle_offset()
        pro2 = Popen(f"imodtrans  -ry {tilt_angle} -tz 187.5 ampa_positions/{context.scene.FA_position_name}_flat.mod ampa_positions/{context.scene.FA_position_name}.mod", stdout=PIPE, stderr=PIPE, shell=True)
        pro2.communicate()
        
        # Read transformed coordinats back in and use to recalc angles... A bit roundabout way, but will be consistent with IMOD
        trans_model = imodmodel.read(f"ampa_positions/{context.scene.FA_position_name}.mod")
        # Iterate over contours (column c in trans_model dataframe)
        contour_grouped = trans_model.groupby('contour_id')

       
        trans_relion_results = []
        for contour, points in contour_grouped:
            # Get the first point of the contour
            point = points.iloc[0][['x','y','z']].to_numpy()
            # Get the coordinates
            trans_coordinates = point * 4

            trans_center_point = points.iloc[1][['x','y','z']].to_numpy() - point
            trans_aunp_point = points.iloc[2][['x','y','z']].to_numpy() - point
            rot, angle = R.align_vectors([[0, 0, 20],[60,0,200]], [trans_center_point,trans_aunp_point])
            euler = rot.as_euler("ZYZ", degrees=True)
            trans_relion_results.append({
                            "rlnCoordinateX": trans_coordinates[0],
                            "rlnCoordinateY": trans_coordinates[1],
                            "rlnCoordinateZ": trans_coordinates[2],
                            "rlnOriginXAngst": 0,
                            "rlnOriginYAngst": 0,
                            "rlnOriginZAngst": 0,
                            "rlnAngleRot": euler[0],
                            "rlnAngleTilt": euler[1],
                            "rlnAnglePsi": euler[2],
                            "rlnImagePixelSize": 2.5,
                            "rlnMicrographName": path,
                            "rlnOpticsGroup": 1,
                        })
        starfile.write({
                    'particles': pd.DataFrame(trans_relion_results), 
                    'optics': pd.DataFrame([
                        {'rlnOpticsGroup':1}
                        ])
                }, f"ampa_positions/{context.scene.FA_position_name}.star")

        

        return {'FINISHED'}
    

# ------------------------------------------------------------------------
#    Panel in Object Mode
# ------------------------------------------------------------------------

class OBJECT_PT_CustomPanel(Panel):
    bl_label = 'Finding AMPA'
    bl_idname = 'OBJECT_PT_CustomPanel'
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = 'scene'
    bl_order = 0
    bl_options = {'HEADER_LAYOUT_EXPAND'}
    bl_ui_units_x = 0   


    def draw(self, context):
        layout = self.layout
        obj = context.object

       

        col = layout.column(align=True)
        row_import = col.row()
        row_import.prop(context.scene, 'FA_position_name')
        col.operator(SaveAMPA.bl_idname, text="Save AMPA position", icon="CONSOLE")
        col.operator(SavePRE.bl_idname, text="Save presynaptic membrane", icon="CONSOLE")
        col.operator(SavePOST.bl_idname, text="Save postsynaptic membrane", icon="CONSOLE")
        row_genmem = col.row()
        row_genmem.prop(context.scene, 'FA_membrane_name')
        col.operator(SaveMEM.bl_idname, text="Save generic Membrane", icon="CONSOLE")
        col.operator(ShowDistance.bl_idname, text="Show distance", icon="CONSOLE")

        layout.separator()

# ------------------------------------------------------------------------
#    Registration
# ------------------------------------------------------------------------

def register():
    bpy.utils.register_class(OBJECT_PT_CustomPanel)
    bpy.utils.register_class(SaveAMPA)
    bpy.utils.register_class(SavePRE)
    bpy.utils.register_class(SavePOST)
    bpy.utils.register_class(SaveMEM)
    bpy.utils.register_class(ShowDistance)


def unregister():
    bpy.utils.unregister_class(OBJECT_PT_CustomPanel)
    bpy.utils.unregister_class(SaveAMPA)
    bpy.utils.unregister_class(SavePRE)
    bpy.utils.unregister_class(SavePOST)
    bpy.utils.unregister_class(SaveMEM)
    bpy.utils.unregister_class(ShowDistance)

register()

import molecularnodes as mn


import sys
from pathlib import Path
argv = sys.argv

path = Path(argv[argv.index("--tomogram") + 1]).absolute()
should_setup = True if "--setup" in argv else False
#tomo = mn.io.density.load(path, invert=True)
#bpy.ops.object.modifier_remove(modifier="MolecularNodes")
#tomo.object.hide_set(True)
# Move the tomo object to a collection
#slice_object = bpy.data.objects['Slice']
#slice_object.modifiers[0]["Socket_2"] = tomo.object
if  should_setup:

    bpy.context.evaluated_depsgraph_get().update()
    
    # Load existing picks
    picks = Path("aunps").glob("*.txt")
    for pick in picks:
        coords = np.loadtxt(pick)/10.0
        new_sp_object = mn.blender.obj.create_object(coords, collection=bpy.data.collections['SavedPicks'], name=pick.stem)
        #bpy.data.collections['SavedPicks'].objects.link(new_sp_object)
    membrain = Path("membrain") / f"{path.name[:-4]}_MemBrain_seg_v10_0.ckpt_segmented_smooth.mrc"
    if membrain.exists():
        membrain = mn.io.density.load(membrain.absolute(), invert=False)
        bpy.data.node_groups["MN_density_NewDensity"].nodes["MN_density_style_surface"].inputs[1].default_value = 125
        #bpy.ops.object.modifier_remove(modifier="MolecularNodes")
        membrain.object.hide_set(True)
        membranes_object = bpy.data.objects['Membranes']
        membranes_object.modifiers[0]["Socket_2"] = membrain.object
        bpy.context.evaluated_depsgraph_get().update()
        membranes_object.update_tag()

    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas: # iterate through areas in current screen
            if area.type == 'VIEW_3D':
                for space in area.spaces: # iterate through spaces in current VIEW_3D area
                    if space.type == 'VIEW_3D': # check if space is a 3D view
                        space.shading.type = 'MATERIAL'

    membranes_object = bpy.data.objects['Membranes']                    
    bpy.context.view_layer.objects.active = membranes_object
    bpy.ops.object.modifier_apply(modifier="GeometryNodes")
    bpy.ops.object.mode_set(mode='EDIT')

    bpy.ops.mesh.separate(type='LOOSE')
    bpy.ops.object.mode_set(mode='OBJECT')
    #DESELECT ALL
    bpy.ops.object.select_all(action='DESELECT')
    # Add ann_dens to Membrane object
    # Apply mem modifier
    # Split membranes
