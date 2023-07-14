import torch
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16).to("cuda")
pipe.unet.load_attn_procs("path-to-save-model", weight_name="pytorch_custom_diffusion_weights.bin")
pipe.load_textual_inversion("path-to-save-model", weight_name="<new1>.bin")
prompt = "A photo of <new1> cat"
#prompt = "watercolor painting of <new1> cat wearing sunglasses"
#prompt = "watercolor painting of <new1> cat running through the ocean"
#prompt = "oilcolor painting of <new1> cat playing with a toy"

image = pipe(
    #"a <new1> cat in New Zealand",
    prompt,
    num_inference_steps=100,
    guidance_scale=6.0,
    eta=1.0,
).images[0]
image.save("./output/"+prompt+".png")