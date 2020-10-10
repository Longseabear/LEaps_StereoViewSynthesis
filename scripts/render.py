import matplotlib.pyplot as plt
import numpy as np
from vispy import scene, io
from vispy.scene import visuals
from vispy.visuals.filters import Alpha
import networkx
from utils.geometry import *

img = plt.imread('../source/my.jpg')
h,w = img.shape[:2]
depth = np.tile((np.arange(0,w))/w,(1,h,1))

canvas = scene.SceneCanvas(bgcolor='black', size=(w*3, h*3))
view = canvas.central_widget.add_view()

view.camera = 'perspective'
view.camera.fov = 60

print(view.camera, type(view.camera))
tr = view.camera.transform

vertice = []
faces = []
colors = []

is_inside = lambda y,x: (y>=0 and y<h and x>=0 and x<w)
four_neighbor = lambda y,x: [(y+1,x),(y,x-1),(y-1,x),(y,x+1)]
valid_neighbor = lambda y,x: [(ny,nx) for ny,nx in four_neighbor(y,x) if is_inside(ny,nx)]
pos2idx = lambda y,x: y*w+x
#
def make_faces(y,x):
    faces = []
    points = four_neighbor(y,x)
    for idx in range(len(points)):
        if is_inside(points[idx][0],points[idx][1]) and is_inside(points[(idx+5)%4][0],points[(idx+5)%4][1]):
            faces.append([pos2idx(y,x),pos2idx(points[idx][0],points[idx][1]),
                          pos2idx(points[(idx+5)%4][0],points[(idx+5)%4][1])])

    return faces

for i in range(h):
    for j in range(w):
        vertice.append([j/w,i/h,10]) #depth[0,i,j]
        colors.append(img[i,j,:]/255)
        faces += make_faces(i,j)
vertice = np.stack(vertice,axis=0)
colors = np.stack(colors, axis=0)
faces = np.stack(faces, axis=0)

print(img.shape,vertice, colors.shape, faces.shape)

mesh = visuals.Mesh(shading=None)
mesh.set_data(vertices=vertice, faces=faces, vertex_colors=colors)
mesh.attach(Alpha(1.0))
view.add(mesh)

tr.translate([0,0,0])
tr.rotate(axis=[1,0,0],angle=180)
view.camera.view_changed()
img = canvas.render()

plt.imshow(img)
plt.show()