import os
import sys
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
from IPython.display import display

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    print("Error: HF_TOKEN not found in .env file")
    sys.exit(1)

client = InferenceClient(provider="auto", api_key=HF_TOKEN)

image = client.text_to_image(
    "Lotus with wisteria and white spider lily",
    model="stabilityai/stable-diffusion-xl-base-1.0",
)

image.save("generated_image.png")
print("Saved: generated_image.png")
display(image)