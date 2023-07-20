import torch
from diffusers import AutoencoderKL, DDPMScheduler, DiffusionPipeline, UNet2DConditionModel
from PIL import Image
import numpy as np
import argparse


parser = argparse.ArgumentParser(description="LoRA training script.")
parser.add_argument(
    "--pretrained_model_name_or_path",
    type=str,
    default=None,
    required=True,
    help="Path to pretrained model or model identifier from huggingface.co/models.",
)
parser.add_argument(
    "--prompt",
    type=str,
    default=None,
    required=True,
    help="Path to pretrained model or model identifier from huggingface.co/models.",
)
parser.add_argument(
    "--scale",
    type=float,
    default=1.0
)
parser.add_argument(
    "--num_samples",
    type=int,
    default=5
)
parser.add_argument(
    "--seed",
    type=int,
    default=42
)

args = parser.parse_args()


if __name__ == "__main__":

    pipeline = DiffusionPipeline.from_pretrained(
            args.pretrained_model_name_or_path, revision=None, torch_dtype=torch.float32
        )
    pipeline = pipeline.to(0)

    # load attention processors and new embedding
    pipeline.unet.load_attn_procs("./logs/cat", weight_name="pytorch_lora_weights.bin")

    # run inference
    generator = torch.Generator(device="cuda")
    generator = generator.manual_seed(args.seed)

    images = pipeline([args.prompt]*args.num_samples, num_inference_steps=30, generator=generator).images
    images = np.hstack([np.array(x) for x in images])
    images = Image.fromarray(images)
    images.save("sample.png")
