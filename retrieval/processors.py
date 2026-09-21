"""Full-frame preprocessing without changing the baseline encoder architecture."""
from typing import Any, Dict

from PIL import Image
from torchvision.transforms import CenterCrop, Compose, Resize


def image_hw(size):
    # Transformers SizeDict exposes get() but is not a dict or Mapping.
    if hasattr(size, "get"):
        height, width = size.get("height"), size.get("width")
        if height is not None and width is not None:
            return int(height), int(width)
        edge = size.get("shortest_edge")
        if edge is None:
            raise ValueError(f"Expected height/width or shortest_edge, got {size!r}")
        size = edge
    if isinstance(size, (tuple, list)):
        return int(size[0]), int(size[1])
    return int(size), int(size)


class FullFrameHFProcessor:
    """Keep dataset geometry separate from the encoder's input tensor size."""
    def __init__(self, processor: Any, canvas_size: Dict[str, int], tensor_hw):
        self.processor = getattr(processor, "image_processor", processor)
        self.size = dict(canvas_size)
        self.tensor_hw = image_hw(tensor_hw)
        self.processor.do_center_crop = False
        self.processor.size = dict(height=self.tensor_hw[0], width=self.tensor_hw[1])

    def __call__(self, images, return_tensors="pt"):
        return self.processor(
            images=images, return_tensors=return_tensors,
            do_center_crop=False, size=dict(height=self.tensor_hw[0], width=self.tensor_hw[1]),
        )


class FullFrameTransformProcessor:
    """Replace resize+center-crop with one full-frame resize; keep normalization."""
    def __init__(self, preprocess, canvas_size: Dict[str, int], tensor_hw=None):
        transforms = list(preprocess.transforms)
        geometry = [item for item in transforms if isinstance(item, (Resize, CenterCrop))]
        if tensor_hw is None:
            if not geometry:
                raise ValueError("Cannot infer encoder input size from preprocessing.")
            tensor_hw = geometry[-1].size
        self.tensor_hw = image_hw(tensor_hw)
        self.size = dict(canvas_size)
        resize = next((item for item in geometry if isinstance(item, Resize)), None)
        kwargs = {}
        if resize is not None:
            kwargs["interpolation"] = resize.interpolation
            kwargs["antialias"] = resize.antialias
        self.preprocess = Compose(
            [Resize(self.tensor_hw, **kwargs)]
            + [item for item in transforms if not isinstance(item, (Resize, CenterCrop))]
        )

    def __call__(self, images: Image.Image, return_tensors="pt"):
        del return_tensors
        return {"pixel_values": self.preprocess(images.convert("RGB")).unsqueeze(0)}
