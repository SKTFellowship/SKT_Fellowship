import torch
from diffusers import AutoencoderKL, DDPMScheduler, DiffusionPipeline, UNet2DConditionModel
from PIL import Image
import numpy as np


pretrained_model_name_or_path = "runwayml/stable-diffusion-v1-5"
weight_dtype = torch.float32
seed = 42
num_images = 4
input_prompt = "a photo of <new1> cat"

pipeline = DiffusionPipeline.from_pretrained(
        pretrained_model_name_or_path, revision=None, torch_dtype=weight_dtype
    )
pipeline = pipeline.to(0)

# load attention processors and new embedding
pipeline.unet.load_attn_procs("./logs/cat", weight_name="pytorch_lora_weights.bin")
pipeline.load_textual_inversion("./logs/cat", weight_name="learned_embeds.bin")

# run inference
generator = torch.Generator(device="cuda")
generator = generator.manual_seed(seed)

images = pipeline([input_prompt]*num_images, num_inference_steps=30, generator=generator).images
images = np.hstack([np.array(x) for x in images])
images = Image.fromarray(images)
images.save("sample.png")
