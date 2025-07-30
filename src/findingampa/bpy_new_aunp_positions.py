import bpy
import numpy as np

from bpy.types import (Panel, Operator)

import sys
sys.path.append('/groups/elferich/Documents/MolecularNodes/')
sys.path.append('/goliath/processing/Gouaux/CJS/GitHub/MolecularNodes')
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

def save_membrane(name):
    collection = bpy.data.collections.get(name)
    if collection:
        selected_objects = collection.objects
    else:
        return {'CANCELLED'}
    if len(selected_objects) == 0:
        return {'CANCELLED'}
    bpy.ops.object.select_all(action='DESELECT')
    for so in selected_objects:
        so.select_set(True)
    filename = f"aunps/{name.lower().replace(' ','')}.glb"
    bpy.ops.export_scene.gltf(
        filepath=filename, 
        use_selection=True  # Export only selected objects
    )  
    bpy.ops.object.select_all(action='DESELECT') 
    for i, so in enumerate(selected_objects):           
        vertices = so.data.vertices
        coords = [v.co for v in vertices]
        filename = f"aunps/{name.lower().replace(' ','')}_{i+1}.txt"
        np.savetxt(filename, np.array(coords)*10.0)
    return {'FINISHED'}

def save_aunps():
    import bpy
    import numpy as np
    from dataclasses import dataclass
    import pandas as pd
    import starfile

    display_object = bpy.data.objects["real_data"]

    depsgraph = bpy.context.evaluated_depsgraph_get()

    mine = display_object.evaluated_get(depsgraph)

    @dataclass
    class AttributeTypeInfo:
        dname: str
        dtype: type
        width: int

    TYPES = {
        key: AttributeTypeInfo(*values)
        for key, values in {
            "FLOAT_VECTOR": ("vector", float, [3]),
            "FLOAT_COLOR": ("color", float, [4]),
            "QUATERNION": ("value", float, [4]),
            "INT": ("value", int, [1]),
            "FLOAT": ("value", float, [1]),
            "INT32_2D": ("value", int, [2]),
            "FLOAT4X4": ("value", float, [4, 4]),
            "BOOLEAN": ("value", bool, [1]),
        }.items()
    }


    data = {}

    for attrs in mine.data.attributes:
        if attrs.name.startswith("."):
            continue
        if attrs.name == "color":
            continue

        print(attrs.name)
        data_type = TYPES[attrs.data_type]
        dim = data_type.width
        n_values = n_att = len(attrs.data)
        for dimension in dim:
            n_values *= dimension
        array = np.zeros(n_values,dtype=data_type.dtype)

        attrs.data.foreach_get(data_type.dname,array)
        if dim == [1]:
            data[attrs.name] = array
        else:
            # return an array with one row per item, even if a 1D attribute. Does this make sense?
            data[attrs.name+"X"] = array.reshape((n_att, *dim))[:,0] * 10.0
            data[attrs.name+"Y"] = array.reshape((n_att, *dim))[:,1] * 10.0
            data[attrs.name+"Z"] = array.reshape((n_att, *dim))[:,2] * 10.0

    df = pd.DataFrame(data)
    print(df)
    starfile.write(df,"aunps/aunps_picked.star")

class Save(Operator):
    """Print object name in Console"""
    bl_idname = "scene.save"
    bl_label = "Save annotations"

    def execute(self, context):
        save_membrane("Presynaptic Membranes")
        save_membrane("Postsynaptic Membranes")
        save_membrane("Synaptic Vesicles")
        save_aunps()
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
       
        col.operator(Save.bl_idname, text="Save Annotation", icon="CONSOLE")
        
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
    
   
    membrain = Path("membrain") / f"{path.name[:-4]}_MemBrain_seg_v10_alpha.ckpt_segmented_smooth.mrc"
    if membrain.exists():
        membrain = mn.entities.density.ui.load(membrain.absolute(), invert=False,name="MemBrain")
        bpy.data.node_groups["MN_density_MemBrain"].nodes["Style Density Surface"].inputs[1].default_value = 125
        #bpy.ops.object.modifier_remove(modifier="MolecularNodes")
        membrain.object.hide_set(True)
        membranes_object = bpy.data.objects['Membranes']
        membranes_object.modifiers[0]["Socket_2"] = membrain.object
        bpy.context.evaluated_depsgraph_get().update()
        membranes_object.update_tag()

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

        bpy.context.evaluated_depsgraph_get().update()

    membrain = Path("membrain") / f"{path.name[:-8]}3DCTF_BIN4_ddw_MemBrain_seg_v10_alpha.ckpt_segmented_smooth.mrc"
    if not membrain.exists():
        membrain = Path("membrain") / f"{path.name[:-8]}3DCTF_BIN4_denoised_MemBrain_seg_v10_alpha.ckpt_segmented_smooth.mrc"
    if membrain.exists():
        membrain = mn.entities.density.ui.load(membrain.absolute(), invert=False,name="MemBrain")
        bpy.data.node_groups["MN_density_MemBrain.001"].nodes["Style Density Surface"].inputs[1].default_value = 125
        #bpy.ops.object.modifier_remove(modifier="MolecularNodes")
        membrain.object.hide_set(True)
        membranes_object = bpy.data.objects['Membranes_denoised']
        membranes_object.modifiers[0]["Socket_2"] = membrain.object
        bpy.context.evaluated_depsgraph_get().update()
        membranes_object.update_tag()
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

        bpy.context.evaluated_depsgraph_get().update()

    tomo_density = mn.entities.density.ui.load(path, invert=False,name="Tomo",setup_nodes=False)
    tomo_display_object = bpy.data.objects['Tomo_Display']
    tomo_display_object.modifiers[0]["Socket_4"] = tomo_density.object
    try:
        picks_star = mn.entities.ensemble.ui.load_starfile(f"aunps/aunp_tm.star",name="AuNP_Template_matching",node_setup=False)
    except KeyError:
        pass
    display_object = bpy.data.objects['Display']
    display_object.modifiers[0]["Socket_3"] = tomo_density.object
    display_object.modifiers[0]["Socket_2"] = picks_star.object
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas: # iterate through areas in current screen
            if area.type == 'VIEW_3D':
                for space in area.spaces: # iterate through spaces in current VIEW_3D area
                    if space.type == 'VIEW_3D': # check if space is a 3D view
                        space.shading.type = 'MATERIAL'

   

    bpy.context.evaluated_depsgraph_get().update()
    tomo_display_object.modifiers[0].show_viewport = False
    #tomo_display_object.modifiers[0].show_viewport = True
    display_object.modifiers[0].show_viewport = False
    display_object.modifiers[0].show_viewport = True
