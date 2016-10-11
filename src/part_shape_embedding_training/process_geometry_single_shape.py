#!/usr/bin/python3
# -*- coding: utf-8 -*-

import bpy
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(BASE_DIR))

shape_file = sys.argv[-3]
shape_synset = sys.argv[-2]
shape_md5 = sys.argv[-1]

output_shape_file = shape_file
if output_shape_file.endswith('.obj'):
    output_shape_file = output_shape_file[:-4]
output_shape_file += '_remeshed.obj'

# Clear default objects
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

bpy.ops.import_scene.obj(filepath=shape_file)

# Join the parts into a single mesh
for ob in bpy.context.scene.objects:
    if ob.type == 'MESH':
        ob.select = True
        bpy.context.scene.objects.active = ob
    else:
        ob.select = False

bpy.ops.object.join()

for ob in bpy.context.scene.objects:
    if ob.type == 'MESH':
        ob.select = True
        bpy.context.scene.objects.active = ob
    else:
        ob.select = False

# Remesh the object
bpy.ops.object.modifier_add(type='REMESH')
bpy.context.object.modifiers["Remesh"].octree_depth = 9
bpy.context.object.modifiers["Remesh"].use_remove_disconnected = False
bpy.ops.object.modifier_apply(apply_as='DATA', modifier="Remesh")

# Decimate mesh
bpy.ops.object.modifier_add(type='DECIMATE')
bpy.context.object.modifiers["Decimate"].decimate_type = 'COLLAPSE'
bpy.context.object.modifiers["Decimate"].ratio = 0.01
bpy.ops.object.modifier_apply(apply_as='DATA', modifier="Decimate")

# Triangulate mesh
bpy.ops.object.modifier_add(type='TRIANGULATE')
bpy.ops.object.modifier_apply(apply_as='DATA', modifier="Triangulate")

# Save processed mesh
bpy.ops.export_scene.obj(filepath=output_shape_file)

