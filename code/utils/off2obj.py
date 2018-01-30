#! /usr/bin/python
# Written by John Bowers
# http://johnsresearch.wordpress.com
# 2009
# You are welcome to use this however you want, this is public domain.

import sys

if len(sys.argv) == 3:
    off_path = sys.argv[1]
    obj_path = sys.argv[2]
else:
    print "USAGE: off2obj.py [path to mesh] [output path]"
    sys.exit(0)

# Class Mesh represents a mesh by a vertex list and a face list
# and has a method loadFromOffFile to load the Mesh data from an
# OFF file.
class Mesh:
    """Class Represents a Mesh by (V, F)"""
    def __init__(self):
	self.verts = []
	self.faces = []
	self.nVerts = 0
	self.nFaces = 0
	self.edges = None
    def writeToObjFile(self, pathToObjFile):
	objFile = open(pathToObjFile, 'w')
	objFile.write("# off2obj OBJ File")
	objFile.write("# http://johnsresearch.wordpress.com\n")
	for vert in self.verts:
	    objFile.write("v ")
	    objFile.write(str(vert[0]))
	    objFile.write(" ")
	    objFile.write(str(vert[1]))
	    objFile.write(" ")
	    objFile.write(str(vert[2]))
	    objFile.write("\n")
	objFile.write("s off\n")
	for face in self.faces:
	    objFile.write("f ")
	    objFile.write(str(face[0]+1))
	    objFile.write(" ")
	    objFile.write(str(face[1]+1))
	    objFile.write(" ")
	    objFile.write(str(face[2]+1))
	    objFile.write("\n")
	objFile.close()
    def loadFromOffFile(self, pathToOffFile):
	#Reset this mesh:
	self.verts = []
	self.faces = []
	self.nVerts = 0
	self.nFaces = 0

	#Open the file for reading:
	offFile = open(pathToOffFile, 'r')
	lines = offFile.readlines()

	#Read the number of verts and faces
	params = lines[1].split()
	self.nVerts = int(params[0])
	self.nFaces = int(params[1])

	#split the remaining lines into vert and face arrays
	vertLines = lines[2:2+self.nVerts]
	faceLines = lines[2+self.nVerts:2+self.nVerts+self.nFaces]

	#Create the verts array
	for vertLine in vertLines:
	    XYZ = vertLine.split()
	    self.verts.append([float(XYZ[0]), float(XYZ[1]), float(XYZ[2])])

	#Create the faces array
	for faceLine in faceLines:
	    XYZ = faceLine.split()
	    self.faces.append((int(XYZ[1]), int(XYZ[2]), int(XYZ[3])))
	    if not(int(XYZ[0]) == 3):
		print "ERROR: This OFF loader can only handle meshes with 3 vertex faces."
		print "A face with", XYZ[0], "vertices is included in the file. Exiting."
		offFile.close()
		sys.exit(0)

	#Cleanup
	offFile.close()
    def edgeList(self):
	if not(self.edges == None):
	    return self.edges
	self.edges = []
	for i in range(0, self.nVerts):
	    self.edges.append([])
	for face in self.faces:
	    i = face[0]
	    j = face[1]
	    k = face[2]
	    if not(j in self.edges[i]):
		self.edges[i].append(j)
	    if not(k in self.edges[i]):
		self.edges[i].append(k)
	    if not(i in self.edges[j]):
		self.edges[j].append(i)
	    if not(k in self.edges[j]):
		self.edges[j].append(k)
	    if not(i in self.edges[k]):
		self.edges[k].append(i)
	    if not(j in self.edges[k]):
		self.edges[k].append(j)
	return self.edges

""" Main Program """

mesh = Mesh()
mesh.loadFromOffFile(off_path)
mesh.writeToObjFile(obj_path)