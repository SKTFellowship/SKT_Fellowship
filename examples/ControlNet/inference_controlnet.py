from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, DiffusionPipeline
import torch
import cv2
from PIL import Image
import numpy as np


pretrained_model_name_or_path = "runwayml/stable-diffusion-v1-5"
canny_path = "/opt/condition/canny/iu/image-061.jpg"
prompt = "a color photo of woman"
num_samples = 4

canny = cv2.imread(canny_path)
canny_image = Image.fromarray(canny)

controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float32)
controlnet = controlnet.to(0)

pipeline = StableDiffusionControlNetPipeline.from_pretrained(
    pretrained_model_name_or_path, controlnet=controlnet, torch_dtype=torch.float32
)
pipeline = pipeline.to(0)

# run inference
generator = torch.Generator(device="cuda")
generator = generator.manual_seed(42)

images = pipeline([prompt]*num_samples, [canny_image]*num_samples, num_inference_steps=30, generator=generator).images
images = np.hstack([np.array(x) for x in images])

del pipeline
del controlnet

origin_pipe = DiffusionPipeline.from_pretrained(
        pretrained_model_name_or_path, revision=None, torch_dtype=torch.float32
    )
origin_pipe = origin_pipe.to(0)

origins = origin_pipe([prompt] * num_samples, num_inference_steps=30, generator=generator).images
origins = np.hstack([np.array(x) for x in origins])

images = np.vstack([origins, images])
images = Image.fromarray(images)
images.save("sample.png")

