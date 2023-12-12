# Stereo View Synthesis Project

## Overview

This is the main code for the prototyping version of the Stereo View Synthesis project. The focus is on implementing "Microsaccadic" rendering techniques using event cameras. The code involves reading stereo images, disparity maps, and generating a layered depth image (LDI) for creating an immersive 3D view.

## Project Preview

![Stereo View Synthesis](https://github.com/Longseabear/LEaps_StereoViewSynthesis/blob/master/output.gif?raw=true)

## Features

- **3D View Generation**: The project employs advanced techniques to synthesize stereo views, creating a sense of depth and dimensionality.

## Getting Started

### For Microsacadic Generation

1. Import necessary libraries:

    ```python
    import matplotlib.pyplot as plt
    import torch
    from dataloader.sceneflow_dataloader import *
    from dataloader.ldi import *
    from config.config_utils import *
    import utils.geometry as geometry_helper
    import numpy as np
    import time
    ```

2. Read stereo images and disparity maps:

    ```python
    left, right, left_disp, right_disp = sceneflow_dataloader.read_stereo("source/sceneflow/image/left/0006.png"
                                                                      ,"source/sceneflow/image/right/0006.png"
                                                                      , "source/sceneflow/depth/left/0006.pfm"
                                                                      , "source/sceneflow/depth/right/0006.pfm")
    ```

3. Initialize a video writer:

    ```python
    video_maker = VideoWriter('test2.avi', fps=20)
    ```

4. Load configuration and initialize rotation and translation matrices:

    ```python
    config = Config.from_yaml('config/train.yaml')
    R, T = geometry_helper.get_identity_rotation(), geometry_helper.get_identity_transform()
    ```

5. Initialize Layered Depth Image (LDI):

    ```python
    H, W = left.shape[:2]
    layered_depth_image = LDI.make_LDI_from_config(config.LDI)
    ```

6. Set mesh from left image and disparity map:

    ```python
    layered_depth_image.set_mesh_from_image(left, left_disp)
    ```

7. Disunite discontinuities based on a disparity threshold:

    ```python
    layered_depth_image.disunite_discontinuities(config.disp_threshold)
    ```

8. Merge mesh from right image and disparity map:

    ```python
    layered_depth_image.merge_mesh_from_image(right, right_disp, 1)
    ```

9. Set render information:

    ```python
    layered_depth_image.set_render_infos()
    ```

10. Create microsaccadic movement paths:

    ```python
    paths = make_sacaddes_movement(config.camera_path, max(left_disp.max(), right_disp.max()), 1)
    ```

11. Render images for each microsaccadic movement and write to video:

    ```python
    for R, T in paths:
        img = layered_depth_image.render(R, T)
        video_maker.write_image(img[:,:,:3])
    ```

12. Finish video creation:

    ```python
    video_maker.finish()
    ```

# 3D Mesh Visualization using Vispy

## Overview

This script demonstrates the creation and visualization of a 3D mesh using the Vispy library. The mesh is constructed from an input image, where each pixel in the image corresponds to a vertex in the mesh. Faces are generated to create a mesh representation, and the resulting 3D scene is displayed using Vispy's scene module.

## Requirements

Make sure you have the required libraries installed. You can install them using:

```bash
pip install matplotlib numpy vispy
```

## Usage

1. Clone the repository:

    ```bash
    git clone https://github.com/your-username/your-repository.git
    cd your-repository
    ```

2. Run the main script:

    ```bash
    python main_script.py
    ```

## For stereo view synthesis

1. Import necessary libraries:

    ```python
    import matplotlib.pyplot as plt
    import numpy as np
    from vispy import scene
    from vispy.scene import visuals
    from vispy.visuals.filters import Alpha
    from utils.geometry import *
    ```

2. Load an image and create depth values:

    ```python
    img = plt.imread('../source/my.jpg')
    h, w = img.shape[:2]
    depth = np.tile((np.arange(0, w))/w, (1, h, 1))
    ```

3. Create a Vispy canvas and views:

    ```python
    canvas = scene.SceneCanvas(bgcolor='black', size=(w*3, h*3))
    grid = canvas.central_widget.add_grid()
    view = canvas.central_widget.add_view()
    left_view = grid.add_view(name='left_view', border_color='yellow')
    ```

4. Create a mesh and set its data:

    ```python
    mesh = visuals.Mesh(shading=None)
    mesh.set_data(vertices=vertice, faces=faces, vertex_colors=colors)
    mesh.attach(Alpha(1.0))
    ```

5. Add the mesh to the scene and set camera transformations:

    ```python
    view.add(mesh)
    tr = view.camera.transform
    tr.translate([0, 0, 0])
    tr.rotate(axis=[1, 0, 0], angle=180)
    view.camera.view_changed()
    ```

6. Render and display the 3D scene:

    ```python
    img = canvas.render()
    plt.imshow(img)
    plt.show()
    ```

Feel free to customize and experiment with the code to suit your specific requirements. :)


