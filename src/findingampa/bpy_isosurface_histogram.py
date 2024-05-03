import bpy
import numpy as np

import molecularnodes as mn
try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    mn.pkg.install_package("matplotlib")
    import matplotlib.pyplot as plt

from bpy.types import (Panel, Operator)

# ------------------------------------------------------------------------
#    Operators
# ------------------------------------------------------------------------

class PlotHistogram(Operator):
    """Print object name in Console"""
    bl_idname = "scene.plot_histogram"
    bl_label = "Plot Histogram"

    def execute(self, context):
        auto_picks_object = bpy.data.objects['Coordinates.001']
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
        volumes = mn.blender.obj.get_attribute(saved_picks_object, "volume")
        logbins = np.logspace(np.log10(0.001),np.log10(1000000000),30)
        cnts, values, bars = plt.hist(volumes, bins=logbins)
        plt.xscale('log')

        for i, (cnt, value, bar) in enumerate(zip(cnts, values, bars)):
            if value < np.power(10, 3.5):
                bar.set_facecolor('grey')
                continue
            if value < np.power(10, 4.5):
                bar.set_facecolor('green')
                continue
            if value < np.power(10, 5.5):
                bar.set_facecolor('blue')
                continue
            bar.set_facecolor('red')
        # X label cubic angstrom (use correct symbol)
        plt.xlabel('Volume (\u212B^3)')
        plt.ylabel('Count')
        # Change colors for certain histogram bars
        plt.show()
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
        col.operator(PlotHistogram.bl_idname, text="Plot Histogram", icon="CONSOLE")

        layout.separator()

# ------------------------------------------------------------------------
#    Registration
# ------------------------------------------------------------------------

def register():
    bpy.utils.register_class(OBJECT_PT_CustomPanel)
    bpy.utils.register_class(PlotHistogram)

def unregister():
    bpy.utils.unregister_class(OBJECT_PT_CustomPanel)
    bpy.utils.unregister_class(PlotHistogram)

register()




import sys
from pathlib import Path
argv = sys.argv

path = Path(argv[argv.index("--tomogram") + 1]).absolute()
tomo = mn.io.density.load(path, invert=True)
bpy.ops.object.modifier_remove(modifier="MolecularNodes")
tomo.object.hide_set(True)
# Move the tomo object to a collection

isosurface_object = bpy.data.objects['Isosurface']
isosurface_object.modifiers[0]["Socket_2"] = tomo.object
isosurface_object.modifiers[0]["Socket_3"] = float(argv[argv.index("--threshold") + 1])
bpy.context.evaluated_depsgraph_get().update()


#membrain = Path("membrain") / f"{path.name[:-4]}_MemBrain_seg_v10_alpha.ckpt_segmented.mrc"
#print(membrain)
#if membrain.exists():
#    membrain = mn.io.density.load(membrain.absolute(), invert=False)
    #bpy.ops.object.modifier_remove(modifier="MolecularNodes")
    #membrain.object.hide_set(True

for window in bpy.context.window_manager.windows:
    for area in window.screen.areas: # iterate through areas in current screen
        if area.type == 'VIEW_3D':
            for space in area.spaces: # iterate through spaces in current VIEW_3D area
                if space.type == 'VIEW_3D': # check if space is a 3D view
                    space.shading.type = 'MATERIAL'


isosurface_object.modifiers["GeometryNodes"].show_viewport = False
isosurface_object.modifiers["GeometryNodes"].show_viewport = True

