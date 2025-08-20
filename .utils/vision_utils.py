
import torch
import torchvision
from typing import List
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
# Visualize a random image
def vis_image(image,
              label: int = None,
              class_names: List[str] = None,
              mode: str = "gray"):
  """Displays an image using pyplot subplots

  Args:
    image_list (torch.Tensor or PIL Image):
    label (int = None): the integer label for the image
    class_names (List[str] = None): A list of string names with indexes coresponding to the label int
    mode (str = "gray"): mode of display options are ["gray", "color"]

  Returns:
    None
  """
  # if image is of PIL Image type
  if isinstance(image, Image.Image):
    # Convert to a numpy array
    image = np.array(image)
  elif isinstance(image, torch.Tensor):
    # assumes torch
    if mode == "color":
      image = image.permute(1, 2, 0).numpy()
  else:
    raise TypeError(f"image is expected to be of type PIL.Image.Image or torch.tensor")

  plot_title = "Image"
  if class_names and label:
    plot_title = class_names[label]

  # plot the figure with matplot lib
  plt.figure(figsize=(10,7))
  if mode == "gray":
    plt.imshow(image, cmap=mode)
  else:
    plt.imshow(image)
  plt.title(plot_title)
  plt.axis(False)

def apply_transforms(examples, my_transform):
  """Applied to hugging face datasets using dataset.with_transform(lambda x: apply_transforms(x, image_transforms))
     Made for use with pytorch torchvision.transforms library

  Args:
    examples: input by the dataset
    my_transform: a torchvision.transforms object

  Returns:
    A list of transformed data that will be applied to the dataset
  """
  images = examples["image"]
  if isinstance(images, list):
    examples["image"] =  [my_transform(image) for image in images]
  else:
    examples["image"] =  my_transform(images)

  return examples
