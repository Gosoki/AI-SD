import gradio as gr
import requests
import base64
from PIL import Image
import io
import os
import time

def generate_image(
    prompt, 
    negative_prompt, 
    steps=20, 
    sampler="Euler a", 
    width=512, 
    height=512, 
    cfg_scale=7.0, 
    seed=-1,
    api_url="http://10.36.63.169:7860/sdapi/v1/txt2img"
):
    """Generate an image using the Stable Diffusion API"""
    
    payload = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "steps": steps,
        "sampler_index": sampler,
        "width": width,
        "height": height,
        "cfg_scale": cfg_scale,
        "seed": seed
    }
    
    try:
        response = requests.post(api_url, json=payload)
        response.raise_for_status()
        result = response.json()
        
        # Convert the base64 string to an image
        image_data = base64.b64decode(result['images'][0])
        image = Image.open(io.BytesIO(image_data))
        
        # Save the image to file
        os.makedirs("outputs", exist_ok=True)
        timestamp = int(time.time())
        filename = f"outputs/image_{timestamp}.png"
        image.save(filename)
        
        return image
    except Exception as e:
        return None

# Create a simpler UI using Interface
iface = gr.Interface(
    fn=generate_image,
    inputs=[
        gr.Textbox(label="Positive Prompt", placeholder="Enter what you want in the image", lines=3),
        gr.Textbox(label="Negative Prompt", placeholder="Enter what you don't want in the image", lines=3),
        gr.Slider(minimum=10, maximum=50, value=20, step=1, label="Steps"),
        gr.Dropdown(
            choices=["Euler a", "Euler", "LMS", "Heun", "DPM2", "DPM2 a", "DPM++ 2S a", "DPM++ 2M", "DPM++ SDE"],
            value="Euler a",
            label="Sampler"
        ),
        gr.Slider(minimum=256, maximum=1024, value=512, step=64, label="Width"),
        gr.Slider(minimum=256, maximum=1024, value=512, step=64, label="Height"),
        gr.Slider(minimum=1.0, maximum=15.0, value=7.0, step=0.5, label="CFG Scale"),
        gr.Number(value=-1, label="Seed (-1 for random)", precision=0),
        gr.Textbox(label="API URL", value="http://10.36.63.169:7860/sdapi/v1/txt2img", lines=1)
    ],
    outputs=gr.Image(type="pil", label="Generated Image"),
    title="Stable Diffusion Image Generator",
    description="Generate images using Stable Diffusion API by providing prompts and parameters"
)

if __name__ == "__main__":
    iface.launch(share=True) 