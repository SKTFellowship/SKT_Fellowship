import torch
import diffusers
from diffusers import DiffusionPipeline
from utils import get_indicator
import os
import numpy as np 
import cv2
from einops import rearrange
from torchvision.utils import make_grid
from PIL import Image
import copy


file = open("prompt.txt", "r")

#######################################################
####                  Custom part                 #####
#######################################################
prompt = file.readline() #default = A photo of <itm>
class_name = "dog"
model_dir = "exp2_model"
output_path = os.path.join(model_dir,'output')
stable_version = '1-5'
scheduler = "EulerAncestralDiscreteScheduler"
use_custom_data = True
safety = False # prevent to NSFW
gen_num = 5
gen_resolution = 512
concat_size = 196
compare_num = 2 #stable & custom
USING_SCHEDULER = True
#######################################################

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


if not os.path.exists(output_path):
    os.makedirs(output_path)

#change prompt indicator <itm> to <new1> cat
indicator = get_indicator(model_dir=model_dir)
prompt = prompt.replace("<itm>", indicator+" "+class_name)

stable_prompt = prompt.replace("<itm>",class_name)

#######################################################
####                 Custom prompt                #####
#######################################################

# prompt  = 'A <new1> dog '
# stable_prompt  = 'Painting of dog at a beach by artist claude monet'


'''
please check the url (https://huggingface.co/docs/diffusers/v0.18.2/en/api/pipelines/stable_diffusion/text2img#diffusers.StableDiffusionPipeline)
'''
if stable_version == '1-4':
    pipe = DiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16).to("cuda")
elif stable_version == '1-5':
    pipe = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16).to("cuda")
    
else:
    print("[Error] : please put the stable-diffusion version '1-4' / '1-5'")

stable = copy.deepcopy(pipe)



if USING_SCHEDULER:
    pipe.scheduler = scheduler_dict[scheduler]
    stable.scheduler = scheduler_dict[scheduler]

if safety == False:
    pipe.safety_checker = None
    pipe.requires_safety_checker = False
    stable.safety_checker = None
    stable.requires_safety_checker = False

if use_custom_data: #using origin Stable diffusion without finetuning
    print("Custom diffusion using for finetuning")
    pipe.unet.load_attn_procs(model_dir, weight_name="pytorch_custom_diffusion_weights.bin") 
    pipe.load_textual_inversion(model_dir, weight_name="<new1>.bin") 
else:
    print("Stable diffusion Without using finetuning")


if not os.path.exists(os.path.join(output_path,prompt)):
    os.makedirs(os.path.join(output_path,prompt))

#genrater
stable_lst = []
custom_lst = []


for count in range(gen_num):
    image = stable(
        stable_prompt,
        num_inference_steps=50, #generally use 50
        guidance_scale=6.0, #text 반영율 6~7
        eta=1.0, 
    ).images[0]
    print("Complete to generate [%d / %d] stable images" % (count+1,gen_num))
    if not os.path.exists(os.path.join(output_path,prompt,"stable")):
        os.makedirs(os.path.join(output_path,prompt,"stable"))
    image.save(os.path.join(output_path,prompt,"stable",str(count)+".png")) 
    stable_lst.append(image.resize((concat_size,concat_size)))



for count in range(gen_num):
    image = pipe(
        prompt,
        num_inference_steps=50, #generally use 50
        guidance_scale=6.0, #text 반영율 6~7
        eta=1.0, 
    ).images[0]
    print("Complete to generate [%d / %d] custom images" % (count+1,gen_num))
    image.save(os.path.join(output_path,prompt,str(count)+".png")) 
    custom_lst.append(image.resize((concat_size,concat_size)))


new_im = Image.new('RGB', (concat_size*gen_num,concat_size*compare_num)) #concat palette
idx = 0
for elem in stable_lst:
    new_im.paste(elem, (idx,0))
    idx += concat_size

idx = 0
for elem in custom_lst:
    new_im.paste(elem, (idx,concat_size))
    idx += concat_size
new_im.save(os.path.join(output_path,prompt,prompt+".png"))
print("please check the directory :",os.path.join(os.getcwd(),output_path,prompt,prompt))
#concat_img = cv2.hconcat([custom_lst[0],custom_lst[1],custom_lst[2],custom_lst[3],custom_lst[4]])
#cv2.imwrite(os.path.join(output_path,prompt+".png", concat_img))

#image.save(os.path.join(output_path,prompt+".png")) 
#concat_image.save(os.path.join(output_path,prompt+".png")) 
file.close()