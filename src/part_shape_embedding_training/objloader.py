def MTL(path, file):
    contents = {}
    mtl = None
    filename = path + file

    for line in open(filename, "r"):
        if line.startswith('#'): continue
        values = line.split()
        if not values: continue
        if values[0] == 'newmtl':
            mtl = contents[values[1]] = {}
        elif mtl is None:
            raise ValueError, "mtl file doesn't start with newmtl stmt"
        elif values[0] == 'map_Kd':
            # load the texture referred to by this declaration
            mtl[values[0]] = values[1]
        else:
            mtl[values[0]] = map(float, values[1:])
    return contents


class OBJ:
    def __init__(self, path, file):
        """Loads a Wavefront OBJ file. """
        self.vertices = []
        self.normals = []
        self.texcoords = []
        self.faces = []

        filename = path + file

        material = None
        for line in open(filename, "r"):
            if line.startswith('#'): continue
            values = line.split()
            if not values: continue
            if values[0] == 'v':
                v = map(float, values[1:4])
                self.vertices.append(v)
            elif values[0] == 'vn':
                v = map(float, values[1:4])
                self.normals.append(v)
            elif values[0] == 'vt':
                self.texcoords.append(map(float, values[1:3]))
            elif values[0] in ('usemtl', 'usemat'):
                material = values[1]
            elif values[0] == 'mtllib':
                self.mtlName = values[1]
                self.mtl = MTL(path, values[1])
            elif values[0] == 'f':
                face = []
                texcoords = []
                norms = []
                for v in values[1:]:
                    w = v.split('/')
                    face.append(int(w[0]))
                    if len(w) >= 2 and len(w[1]) > 0:
                        texcoords.append(int(w[1]))
                    else:
                        texcoords.append(0)
                    if len(w) >= 3 and len(w[2]) > 0:
                        norms.append(int(w[2]))
                    else:
                        norms.append(0)
                self.faces.append((face, norms, texcoords, material))


    def saveOBJ(self, path, file):
        """Saves a Wavefront OBJ file. """

        filename = path + file
        f = open(filename, 'w')

        if self.mtlName:
            f.write('mtllib' + ' ' + self.mtlName + '\n')

        n_v = len(self.vertices)
        for i in range(0, n_v):
            f.write('v' + ' ' + str(self.vertices[i][0]) + ' ' + str(self.vertices[i][1]) + ' ' + str(self.vertices[i][2]) + '\n')

        n_vn = len(self.normals)
        for i in range(0, n_vn):
            f.write('vn' + ' ' + str(self.normals[i][0]) + ' ' + str(self.normals[i][1]) + ' ' + str(self.normals[i][2]) + '\n')

        n_vt = len(self.texcoords)
        for i in range(0, n_vt):
            f.write('vt' + ' ' + str(self.texcoords[i][0]) + ' ' + str(self.texcoords[i][1]) + '\n')

        n_mtl = len(self.mtl)
        n_faces = len(self.faces)
        for mtl_id in range(0, n_mtl):
            f.write('usemtl' + ' ' + self.mtl.keys()[mtl_id] + '\n')
            for i in range(0, n_faces):
                a = self.faces[i][0]
                b = self.faces[i][1]
                c = self.faces[i][2]
                material = self.faces[i][3]

                if self.mtl.keys()[mtl_id] == material:
                    if b[0] == 0:
                        f.write('f' + ' ' + str(a[0])+'/'+str(c[0]) + ' ' + str(a[1])+'/'+str(c[1]) + ' ' + str(a[2])+'/'+str(c[2]) + '\n')
                    else:
                        f.write('f' + ' ' + str(a[0])+'/'+str(b[0])+'/'+str(c[0]) + ' ' + str(a[1])+'/'+str(b[1])+'/'+str(c[1]) + ' ' + str(a[2])+'/'+str(b[2])+'/'+str(c[2]) + '\n')

        f.close()

