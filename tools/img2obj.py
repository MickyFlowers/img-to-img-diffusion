import os
import trimesh
import PIL.Image
import numpy as np
from trimesh.visual import TextureVisuals
from trimesh.visual.material import SimpleMaterial


def image_to_obj(
    image_path: str, 
    save_folder: str = None, 
    scale = 1.0, 
    vis: bool = False, 
):
    image = PIL.Image.open(image_path)
    H, W = image.height, image.width

    vertices = np.array([
        [-1, -1, 0], [1, -1, 0],
        [-1, 1, 0], [1, 1, 0]
    ]).astype(np.float64) / 2 * scale

    vertices[:, 0] *= W
    vertices[:, 1] *= H
    vertices /= np.sqrt(H*W)

    faces = np.array([[0, 1, 2], [2, 1, 3]])
    material = SimpleMaterial(
        image=image,
        diffuse=None,
        ambient=None,
        specular=None
    )

    texture = TextureVisuals(
        uv=np.array([[0, 0], [1, 0], [0, 1], [1, 1]]),
        material=material
    )

    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        visual=texture
    )

    if save_folder is not None:
        image_name = os.path.split(image_path)[-1]
        image_name_no_ext = os.path.splitext(image_name)[0]
        export_folder = os.path.join(save_folder, image_name_no_ext)
        os.makedirs(export_folder, exist_ok=True)
        save_path = os.path.join(export_folder, "model.obj")
        mesh.export(save_path)
    else:
        save_path  = None

    if vis:
        mesh.show()
    
    return mesh, save_path


if __name__ == "__main__":
    import glob

    here = os.path.dirname(__file__)
    image_paths = glob.glob(f"{here}/images/*.png")

    for image_path in image_paths:
        image_to_obj(image_path, f"{here}/meshes")

