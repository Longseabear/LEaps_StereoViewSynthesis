
import math
import numpy as np
import utils.geometry as geometry_helper

def get_neighbors(mesh, node):
    return [*mesh.neighbors(node)]

def generate_face(mesh, disp_threshold=None):
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

    for node in nodes:
        cur_id = nodes[node]['vertex_id']

        neighbor_nodes = get_neighbors(mesh, node)
        distance = {}
        da = [(-1,0), (0,-1), (1,0), (0,1)]
        db = [(0,-1), (1,0), (0,1), (-1,0)]
        for y, x in da:
            distance[(y,x)] = []

        for ne in neighbor_nodes:
            distance_key = (np.sign(ne[0]-node[0]), np.sign(ne[1]-node[1]))
            distance[distance_key].append((ne, nodes[ne]['vertex_id']))

        for a_, b_ in zip(da, db):
            for node_a, idx_a in distance[(a_[0], a_[1])]:
                for node_b, idx_b in distance[(b_[0], b_[1])]:
                    faces.append([idx_a, cur_id, idx_b])

    return faces

def mesh_to_render_points(mesh, K, inv_K, depth_scale=10):
    nodes = mesh.nodes
    nodes_points = []
    nodes_colors = []
    vertex_id = 0

    for node in nodes:
        x,y,z = nodes[node]['points']

        nodes_points.append(geometry_helper.pix2cam((x,y,z), inv_K))
        nodes_colors.append(nodes[node]['color'])
        nodes[node]['vertex_id'] = vertex_id
        vertex_id += 1

    faces_point = generate_face(mesh)

    return nodes_points, nodes_colors, faces_point

def mesh_to_render_points_for_stereo(mesh_left, mesh_right, K, inv_K):
    nodes_points = []
    nodes_colors = []
    vertex_id = 0

    for node in mesh_left.nodes:
        x,y,z = mesh_left.nodes[node]['points']

        nodes_points.append(geometry_helper.pix2cam((x,y,z), inv_K))
        nodes_colors.append(mesh_left.nodes[node]['color'])
        mesh_left.nodes[node]['vertex_id'] = vertex_id
        vertex_id += 1

    for node in mesh_right.nodes:
        x,y,z = mesh_right.nodes[node]['points']

        node_point = geometry_helper.pix2cam((x,y,z), inv_K)
        node_point[0] += 1
        nodes_points.append(node_point)
        nodes_colors.append(mesh_right.nodes[node]['color'])
        mesh_right.nodes[node]['vertex_id'] = vertex_id
        vertex_id += 1

    faces_point = generate_face(mesh_left)
    faces_point += generate_face(mesh_right)

    return nodes_points, nodes_colors, faces_point


def init_mesh(ldi, image, disp, base_line=1):
    """
    Initialize the mesh based on the image and disparity.

    @param ldi: Networkx graph
    @param image: image [H,W,3]
    @param disp: disparity [H,W,1]
    @param base_line: two camera base line length. default: 1
    @return: mesh
    """
    mesh_nodes = ldi.mesh.nodes

    h, w = ldi.mesh.graph['H'], ldi.mesh.graph['W']
    depth = (ldi.virtual_camera.focal_length_pix * base_line) / disp

    for y in range(h):
        for x in range(w):
            ldi.mesh.add_node((y, x, 0),
                              points=(x, y, -depth[y,x]),
                              disp=disp[y, x],
                              color=image[y, x])
            ldi.pixel_layer_nums[y, x] = 1

    for node in mesh_nodes:
        x,y,d = mesh_nodes[node]['points']
        two_nes = [ne for ne in [(y + 1, x), (y, x + 1)] if
                   ne[0] < h and ne[1] < w]
        [ldi.mesh.add_edge((ne[0], ne[1], 0), (y, x, 0)) for ne in two_nes]
    return ldi.mesh

def add_node(ldi, y, x, **args):
    ldi.mesh.add_node((y, x, ldi.pixel_layer_nums[y, x]), **args)
    ldi.pixel_layer_nums[y, x] += 1

def merge_mesh(ldi, image, disp, threshold, base_line=1):
    """
    merge the mesh based on the image and disparity using forward warping.
    This method uses disparity as it is. You must check the sign of disparity.
    ex) if right->left, disparity is positive, opposite is negative

    @param ldi: Networkx graph
    @param image: image [H,W,3]
    @param disp: disparity [H,W,1] right -> virtual camera view (forward)
    @param threshold: disparity threshold
    @param base_line: two camera base line length. default: 1

    @return: mesh
    """
    h, w = ldi.mesh.graph['H'], ldi.mesh.graph['W']
    depth = (ldi.virtual_camera.focal_length_pix * base_line) / disp
    nodes = ldi.mesh.nodes

    yy, xx = np.meshgrid(range(h), range(w), indexing='ij')  # w h
    yy, xx = yy.astype(np.float), xx.astype(np.float)
    xx += disp
    next_idx = np.stack([yy, xx], axis=-1)

    modified_info = {}

    """
    functions
    @function is_inside(y,x): coordinate valid check
    @function sum_tuple(a, b): a + b
    """
    is_inside = lambda y, x: x>=0 and x<w and y>=0 and y<h
    elementwise_summation = lambda a, b: (a[0]+b[0], a[1]+b[1], a[2]+b[2])
    elementwise_divide = lambda a, b: (a[0]/b, a[1]/b, a[2]/b)
    new_nodes = []

    for y in range(h):
        for x in range(w):
            dy, dx = next_idx[y,x]
            dy_int, dx_int = int(round(dy)), int(round(dx))

            if not is_inside(dy_int, dx_int):
                continue

            layers_num = ldi.pixel_layer_nums[dy_int, dx_int]
            valid_idx = None
            cur_disp = abs(disp[y,x])
            for i in range(layers_num):
                if abs(nodes[(dy_int, dx_int, i)]['disp']-cur_disp) < threshold:
                    valid_idx = i
                    break
            if valid_idx is None:
                ldi.mesh.add_node((dy_int, dx_int, ldi.pixel_layer_nums[dy_int, dx_int]),
                                  points=(dx, dy, -depth[y, x]),
                                  disp=cur_disp,
                                  color=image[y, x])
                origin_info = ldi.mesh.nodes[(dy_int, dx_int, ldi.pixel_layer_nums[dy_int, dx_int])]

                modified_info[(dy_int, dx_int, ldi.pixel_layer_nums[dy_int, dx_int])] = {
                    'points': origin_info['points'],
                    'color': origin_info['color'],
                    'disp': origin_info['disp'],
                    'counts': 1,
                }
                new_nodes.append((dy_int, dx_int, ldi.pixel_layer_nums[dy_int, dx_int]))
                ldi.pixel_layer_nums[(dy_int, dx_int)] += 1
            else:
                if modified_info.get((dy_int, dx_int, valid_idx)) is None:
                    origin_info = ldi.mesh.nodes[(dy_int, dx_int, valid_idx)]
                    modified_info[(dy_int, dx_int, valid_idx)] = {
                        'points': origin_info['points'],
                        'color': origin_info['color'],
                        'disp': origin_info['disp'],
                        'counts': 1,
                    }
                origin_info = modified_info[(dy_int,dx_int,valid_idx)]

                modified_info[(dy_int, dx_int, valid_idx)].update({
                'points': elementwise_summation(origin_info['points'], (dx, dy, -depth[y,x])),
                'color': image[y, x]+origin_info['color'],
                'disp': cur_disp + origin_info['disp']})
                modified_info[(dy_int, dx_int, valid_idx)]['counts'] += 1

    for node in modified_info.keys():
        info = modified_info[node]
        info['points'] = elementwise_divide(info['points'], info['counts'])
        info['color'] = info['color'] / info['counts']
        info['disp'] = info['disp'] / info['counts']
        info.pop('counts')
        nodes[node].update(info)

    for node in new_nodes:
        y, x = node[0], node[1]
        four_nes = [ne for ne in [(y + 1, x), (y, x + 1), (y - 1, x), (y, x - 1)] if
                    ne[0] < h and ne[1] < w and ne[0] >= 0 and ne[1] >= 0]

        cur_disp = nodes[node]['disp']

        [ldi.mesh.add_edge(node, (ny, nx, idx)) for ny,nx in four_nes for idx in range(ldi.pixel_layer_nums[ny, nx]) if (abs(nodes[(ny, nx, idx)]['disp']-cur_disp) < threshold)]

    return ldi.mesh

def disunite_discontinuities(ldi, threshold):
    nodes = ldi.mesh.nodes
    remove_edge_list = []
    for edge in ldi.mesh.edges:
        if abs(nodes[edge[0]]['disp'] - nodes[edge[1]]['disp']) >= threshold:
            remove_edge_list.append((edge[0], edge[1])) # add removed list

    ldi.mesh.remove_edges_from(remove_edge_list)
    return ldi.mesh
