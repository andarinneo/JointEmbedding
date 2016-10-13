# IMPORT OBJECT LOADER
from objloader import *
import numpy as np
from shutil import copyfile

def paint_mesh_faces(root_path, class_id, model_id, n_parts, part_labels):

    path = root_path + '/' + class_id + '/' + model_id + '/'

    # LOAD OBJECT
    obj = OBJ(path, 'reduced_model_remeshed.obj')


    # 1. Load dense point file for this 3D model

    points3DFile = root_path + '/../PartAnnotation/' + class_id + '/' + 'points' + '/' + model_id + '.pts'
    points3D = []
    for line in open(points3DFile, "r"):
        values = line.split()
        points3D.append([float(values[0]), float(values[1]), float(values[2])])
    n_points3D = len(points3D)


    # 2. Load point labelling

    partPointLabels = []
    for part_id in range(0, n_parts):
        partPointFile = root_path + '/../PartAnnotation/' + class_id + '/' + 'points_label' + '/' + part_labels[part_id] + '/' + model_id + '.seg'

        pointLabels = []
        for line in open(partPointFile, "r"):
            value = int(line[0])
            pointLabels.append(value)

        partPointLabels.append(pointLabels)

    # Also create the label for each point to simplify accessing the data
    points3Dlabels = []
    for point_id in range(0, n_points3D):
        points3Dlabels.append(-1)
        for part_id in range(0, n_parts):
            label = partPointLabels[part_id][point_id]
            if label:
                points3Dlabels[point_id] = part_id
                break


    # Process the 3D model to paint based on the part labelling

    # 3. Change the material file link inside the OBJ model

    mtl_path = root_path + '/../PartAnnotation/part_colors/'

    copyfile(mtl_path + class_id + '.mtl', path + class_id + '.mtl')

    obj.mtlName = class_id + '.mtl'
    obj.mtl = MTL(mtl_path, class_id + '.mtl')

    obj.saveOBJ(path, 'colored_parts.obj')


    # 4. Loop the faces

    counter = 0
    n_faces = len(obj.faces)
    for face in obj.faces:
        # 5. For each face find the point that better fits, (starts counting points from 1, thus -1 to coordinates)
        p1_id = face[0][0]-1
        p2_id = face[0][1]-1
        p3_id = face[0][2]-1

        p1 = np.array([obj.vertices[p1_id][0], obj.vertices[p1_id][1], obj.vertices[p1_id][2]])
        p2 = np.array([obj.vertices[p2_id][0], obj.vertices[p2_id][1], obj.vertices[p2_id][2]])
        p3 = np.array([obj.vertices[p3_id][0], obj.vertices[p3_id][1], obj.vertices[p3_id][2]])

        barycenter = (p1 + p2 + p3)/3

        minDist = 99999999
        minLabel = -1

        for point_id in range(0, len(points3D)):
            point3D = np.array([points3D[point_id][0], points3D[point_id][1], points3D[point_id][2]])
            dist = abs(np.linalg.norm(barycenter - point3D))

            # Some points do not have a proper label (value == -1) so we should not consider them
            if dist < minDist and points3Dlabels[point_id] >= 0:
                minDist = dist
                minLabel = points3Dlabels[point_id]

        # This goes out of the point loop
        obj.faces[counter] = tuple([face[0], face[1], face[2], part_labels[minLabel]])

        counter += 1

        # if (counter%100 == 0):
        #     print 'Iteration {} out of {}...'.format(counter, n_faces)


    # 6. Save modified 3D model
    obj.saveOBJ(path, 'colored_parts.obj')

    print(path)
