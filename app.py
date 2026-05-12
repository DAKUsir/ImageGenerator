import os
import uuid
from datetime import datetime

from flask import Flask, render_template, request, send_file
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN not found in .env")

client = InferenceClient(provider="auto", api_key=HF_TOKEN)

app = Flask(__name__)
OUTPUT_DIR = os.path.join("static", "images")
os.makedirs(OUTPUT_DIR, exist_ok=True)


@app.route("/", methods=["GET", "POST"])
def index():
    image_path = None
    error = None

    if request.method == "POST":
        prompt = request.form.get("prompt", "").strip()
        if not prompt:
            error = "Prompt is required"
        else:
            negative = request.form.get("negative_prompt", "").strip() or None
            model = request.form.get("model", "stabilityai/stable-diffusion-xl-base-1.0")
            steps = int(request.form.get("steps", 25))
            guidance = float(request.form.get("guidance_scale", 7.5))
            width = int(request.form.get("width", 1024))
            height = int(request.form.get("height", 1024))

            params = dict(prompt=prompt, model=model, num_inference_steps=steps, guidance_scale=guidance, width=width, height=height)
            if negative:
                params["negative_prompt"] = negative

            try:
                image = client.text_to_image(**params)
                filename = f"{uuid.uuid4().hex}.png"
                filepath = os.path.join(OUTPUT_DIR, filename)
                image.save(filepath)
                image_path = f"images/{filename}"
            except Exception as e:
                error = str(e)

    return render_template("index.html", image_path=image_path, error=error)


if __name__ == "__main__":
    app.run(debug=True)
