import os

def get_indicator(model_dir) -> str:
    for dir_name in os.listdir(model_dir):
        if 'bin' in dir_name:
            tmp = dir_name.split(".") 
            indicator = tmp[-2]
            if indicator == "pytorch_custom_diffusion_weights":
                pass
            else:
                return indicator
        else:
            pass 