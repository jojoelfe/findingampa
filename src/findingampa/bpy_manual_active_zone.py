import bpy
import numpy as np

from bpy.types import (Panel, Operator)

import sys

sys.path.append(sys.argv[sys.argv.index("--findingampa") + 1])


# ------------------------------------------------------------------------
#    Operators
# ------------------------------------------------------------------------


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



class Save(Operator):
    """Print object name in Console"""
    bl_idname = "scene.save"
    bl_label = "Save annotations"

    def execute(self, context):
        import json
        cube = bpy.data.objects['Cube']
        center = np.array(cube.location) * 10.0
        size = np.array(cube.scale) * 10.0
        cm = np.array(cube.rotation_euler.to_matrix()).T
        try:
            data = json.loads(Path("fa_manual_override.json").read_text())
        except FileNotFoundError:
            data = {}
        try:
            data["manual_active_zones"].append({"center": center.tolist(), "size": size.tolist(), "rotation": cm.tolist()})
        except KeyError:
            data["manual_active_zones"] = [{"center": center.tolist(), "size": size.tolist(), "rotation": cm.tolist()}]
        Path("fa_manual_override.json").write_text(json.dumps(data, indent=4))    
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
       
        col.operator(Save.bl_idname, text="Save Active Zone", icon="CONSOLE")
        
        col.operator(ShowDistance.bl_idname, text="Show distance", icon="CONSOLE")

        layout.separator()

# ------------------------------------------------------------------------
#    Registration
# ------------------------------------------------------------------------

def register():
    bpy.utils.register_class(OBJECT_PT_CustomPanel)
    bpy.utils.register_class(Save)

    bpy.utils.register_class(ShowDistance)


def unregister():
    bpy.utils.unregister_class(OBJECT_PT_CustomPanel)
    bpy.utils.unregister_class(Save)

    bpy.utils.unregister_class(ShowDistance)

register()



import sys
import os
import mathutils
from pathlib import Path
from findingampa.utils.assets import FA_ASSETS
argv = sys.argv


bm = mathutils.Matrix()
center = np.array([100,100,100])
size = np.array([200,200,50])
bpy.ops.mesh.primitive_cube_add(size=1, location= center / 10.0)


cube = bpy.context.active_object

bevel_modifier = cube.modifiers.new(name="Wireframe", type='WIREFRAME')
bevel_modifier.thickness = 0.01



# Append from blend file
bpy.ops.wm.append("EXEC_DEFAULT",directory=os.path.join(str(FA_ASSETS / "blender" / "coordinatesystem.blend"),"Object"),filename="CoordinateSystem",use_recursive=True)

coordinate_system = bpy.data.objects["CoordinateSystem"]
coordinate_system.location =  (-np.array([10,10,10]) / 20.0)
coordinate_system.scale = np.array([1,1,1]) / 30.0

# Set coordinate_system as the active object
bpy.context.view_layer.objects.active = coordinate_system
# Apply modifier
bpy.ops.object.modifier_apply(modifier="GeometryNodes")
#bpy.ops.object.transform_apply(location=True)
#coordinate_system.location = center / 10.0
#coordinate_system.rotation_euler = bm.to_euler()
# Parent coordinate system to cube
coordinate_system.parent = cube
cube.scale = size / 10.0
cube.rotation_euler = np.array([4.71238899230957,0,0])
bpy.context.view_layer.objects.active = cube


