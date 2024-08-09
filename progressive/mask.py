
from os import makedirs
import torch
from PIL import Image

def get_mask(width, height, contr_per_pixel):
    mask = torch.zeros(width * height, dtype=torch.bool)
    visible = contr_per_pixel > 0
    visible_indices = torch.nonzero(visible, as_tuple=True)[0]
    visible_pixels = torch.div(visible_indices, 20, rounding_mode="floor")
    visible_pixels = torch.unique(visible_pixels)
    mask[visible_pixels] = True
    return mask

def save_mask(view, contr_per_pixel, path, name):
    path = path + "\\masks"
    makedirs(path, exist_ok=True)

    w, h = view.image_width, view.image_height
    mask = get_mask(w, h, contr_per_pixel)

    bool_tensor_2d = mask.view(h, w)
    image_tensor = bool_tensor_2d.to(torch.uint8) * 255
    image = Image.fromarray(image_tensor.numpy(), mode="L")
    image.save(f"{path}\\{name}.png")