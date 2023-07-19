import torch
import diffusers
from diffusers import DiffusionPipeline
from utils import get_indicator
import os




file = open("prompt.txt", "r")

#######################################################
####                  Custom part                 #####
#######################################################
prompt = file.readline() #default = A photo of <itm>
class_name = "dog"
model_dir = "path-to-save-model"
output_path = 'output'
stable_version = '1-4'
scheduler = "EulerAncestralDiscreteScheduler"
use_custom_data = True
safety = True # prevent to NSFW
#######################################################


#change prompt indicator <itm> to <new1> cat
indicator = get_indicator(model_dir=model_dir)
prompt = prompt.replace("<itm>", indicator+" "+class_name)

'''
please check the url (https://huggingface.co/docs/diffusers/v0.18.2/en/api/pipelines/stable_diffusion/text2img#diffusers.StableDiffusionPipeline)
'''
if stable_version == '1-4':
    pipe = DiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16).to("cuda")
elif stable_version == '1-5':
    pipe = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16).to("cuda")
else:
    print("[Error] : please put the stable-diffusion version '1-4' / '1-5'")




'''
please check the url (https://huggingface.co/docs/diffusers/v0.18.2/en/using-diffusers/schedulers)
'''
scheduler_dict= {
    "EulerAncestralDiscreteScheduler":diffusers.EulerAncestralDiscreteScheduler(),
    'LMSDiscreteScheduler':diffusers.LMSDiscreteScheduler(),
    'DDIMScheduler':diffusers.DDIMScheduler(),
    'DPMSolverMultistepScheduler': diffusers.DPMSolverMultistepScheduler(),
    'EulerDiscreteScheduler':diffusers.EulerDiscreteScheduler(),
    'PNDMScheduler':diffusers.PNDMScheduler(),
    'DDPMScheduler':diffusers.DDPMScheduler()
}

pipe.scheduler = scheduler_dict[scheduler]

if safety == False:
    pipe.safety_checker = None
    pipe.requires_safety_checker = False

if use_custom_data: #using origin Stable diffusion without finetuning
    print("Custom diffusion using for finetuning")
    pipe.unet.load_attn_procs(model_dir, weight_name="pytorch_custom_diffusion_weights.bin") 
    pipe.load_textual_inversion(model_dir, weight_name="<new1>.bin") 
else:
    print("Stable diffusion Without using finetuning")
    

#genrater
image = pipe(
    prompt,
    num_inference_steps=50, #generally use 50
    guidance_scale=6.0, #text 반영율 6~7
    eta=1.0, 
).images[0]
print(prompt)
image.save(os.path.join(output_path,prompt+".png")) 
file.close()