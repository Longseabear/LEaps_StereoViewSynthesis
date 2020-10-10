
import math
import numpy as np
import utils.geometry as geometry_helper
def get_neighbors(mesh, node):
    return [*mesh.neighbors(node)]

def generate_face(mesh, depth_threshold=None):
    """
    Convert points to faces.

    @param mesh:
        Networkx graph

    @param depth_threshold:
        Connect only when the difference in depth is less than the depth threshold. default: 0

    @return faces
    """
    faces = []
    nodes = mesh.nodes

    valid_disp_fn = lambda zs: True if depth_threshold is None else all([abs(zs[i]-zs[(i+1)%3]) < depth_threshold for i in range(3)])

    for yxz in nodes:
        cur_id = nodes[yxz]['vertex_id']
        cur_disp = nodes[yxz]['disp']
        neighbor_nodes = get_neighbors(mesh, yxz)
        distance = {}
        da = [(-1,0), (0,-1), (1,0), (0,1)]
        db = [(0,-1), (1,0), (0,1), (-1,0)]
        for y, x in da:
            distance[(y,x)] = []

        for ne in neighbor_nodes:
            distance_key = (np.sign(ne[0]-yxz[0]),np.sign(ne[1]-yxz[1]))
            distance[distance_key].append((ne, nodes[ne]['vertex_id'], nodes[ne]['disp']))

        # left_top
        for a_, b_ in zip(da, db):
            for node_a, idx_a, z_a in distance[(a_[0], a_[1])]:
                for node_b, idx_b, z_b in distance[(b_[0], b_[1])]:
                    if valid_disp_fn((z_a, z_b, cur_disp)):
                        faces.append([idx_a, cur_id, idx_b])

    return faces


def mesh_to_render_points(mesh, K, inv_K):
    nodes = mesh.nodes
    nodes_points = []
    nodes_colors = []
    vertex_id = 0

    for yxz in nodes:
        y,x,z = yxz

        nodes_points.append(geometry_helper.pix2cam((x,y,z),inv_K))
        nodes_colors.append(nodes[yxz]['color'])
        nodes[yxz]['vertex_id'] = vertex_id
        vertex_id+=1

    faces_point = generate_face(mesh)

    return nodes_points, nodes_colors, faces_point


def init_mesh(ldi, image, disp, base_line=1):
    """
    Initialize the mesh based on the image and disparity.

    @param mesh: Networkx graph
    @param image: image [H,W,3]
    @param disp: disparity [H,W,1]
    @param base_line: two camera base line length. default: 1

    @return: mesh
    """
    h, w = ldi.mesh.graph['H'], ldi.mesh.graph['W']
    depth = (ldi.camera.focal_length_pix * base_line) / disp
    for y in range(h):
        for x in range(w):
            ldi.mesh.add_node((y, x, -depth[y, x]),
                               disp=disp[y, x],
                               color=image[y, x])
            ldi.z_buffer[(y, x)] = [-depth[y, x]]
    for y, x, d, in ldi.mesh.nodes:
        two_nes = [ne for ne in [(y + 1, x), (y, x + 1)] if
                   ne[0] < h and ne[1] < w]
        [ldi.mesh.add_edge((ne[0], ne[1], ldi.z_buffer[ne][0]), (y, x, d)) for ne in two_nes]
    return ldi.mesh
