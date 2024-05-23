import bpy
import numpy as np

from bpy.types import (Panel, Operator)

# ------------------------------------------------------------------------
#    Operators
# ------------------------------------------------------------------------

class SaveAuto(Operator):
    """Print object name in Console"""
    bl_idname = "scene.save_auto"
    bl_label = "Save AutoPicks"

    def execute(self, context):
        auto_picks_object = bpy.data.objects['Coordinates']
        saved_picks_object = auto_picks_object.copy()
        saved_picks_object.data = auto_picks_object.data.copy()
        saved_picks_object.animation_data_clear()
        saved_picks_collection = bpy.data.collections['SavedPicks']
        # Get number of items in collection
        num_items = len(saved_picks_collection.objects)
        # Set name of saved picks object
        saved_picks_object.name = f"SavedPicks_{num_items+1:03d}"
        saved_picks_collection.objects.link(saved_picks_object)
        bpy.context.view_layer.objects.active = saved_picks_object
        bpy.ops.object.modifier_apply(modifier='GeometryNodes')
        return {'FINISHED'}
    
class WritePicks(Operator):
    """Print object name in Console"""
    bl_idname = "scene.write_picks"
    bl_label = "Write Saved Picks"

    def execute(self, context):
        from subprocess import Popen, PIPE
        # Iterate over objects in SavedPicks collection
        saved_picks_collection = bpy.data.collections['SavedPicks']
        for obj in saved_picks_collection.objects:
            # Get vertex coordinates as numpy array
            vertices = obj.data.vertices
            coords = [v.co for v in vertices]
            coords = np.array(coords)
            np.savetxt(f"aunps/{obj.name}.txt", coords*10.0)
            
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
        col.operator(SaveAuto.bl_idname, text="Save Current Picks", icon="CONSOLE")
        col.operator(WritePicks.bl_idname, text="Write Saved Picks To Disk", icon="CONSOLE")

        layout.separator()

# ------------------------------------------------------------------------
#    Registration
# ------------------------------------------------------------------------

def register():
    bpy.utils.register_class(OBJECT_PT_CustomPanel)
    bpy.utils.register_class(SaveAuto)
    bpy.utils.register_class(WritePicks)

def unregister():
    bpy.utils.unregister_class(OBJECT_PT_CustomPanel)
    bpy.utils.unregister_class(SaveAuto)
    bpy.utils.unregister_class(WritePicks)

register()

import molecularnodes as mn


import sys
from pathlib import Path
argv = sys.argv

path = Path(argv[argv.index("--tomogram") + 1]).absolute()
tomo = mn.io.density.load(path, invert=True)
bpy.ops.object.modifier_remove(modifier="MolecularNodes")
tomo.object.hide_set(True)
# Move the tomo object to a collection
slice_object = bpy.data.objects['Slice']
slice_object.modifiers[0]["Socket_2"] = tomo.object

bpy.context.evaluated_depsgraph_get().update()

# Load existing picks
picks = Path("aunps").glob("*.txt")
for pick in picks:
    coords = np.loadtxt(pick)/10.0
    new_sp_object = mn.blender.obj.create_object(coords, collection=bpy.data.collections['SavedPicks'], name=pick.stem)
    #bpy.data.collections['SavedPicks'].objects.link(new_sp_object)
membrain = Path("membrain") / f"{path.name[:-4]}_MemBrain_seg_v10_alpha.ckpt_segmented_smooth.mrc"
print(membrain)
if membrain.exists():
    membrain = mn.io.density.load(membrain.absolute(), invert=False)
    #bpy.ops.object.modifier_remove(modifier="MolecularNodes")
    #membrain.object.hide_set(True

for window in bpy.context.window_manager.windows:
    for area in window.screen.areas: # iterate through areas in current screen
        if area.type == 'VIEW_3D':
            for space in area.spaces: # iterate through spaces in current VIEW_3D area
                if space.type == 'VIEW_3D': # check if space is a 3D view
                    space.shading.type = 'MATERIAL'

slice_object.modifiers["GeometryNodes"].show_viewport = False
slice_object.modifiers["GeometryNodes"].show_viewport = True


