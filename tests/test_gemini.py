import os
from dotenv import load_dotenv
from google import genai
from PIL import Image

# --------------------
# Load API key (optional if already hardcoded for now)
# --------------------
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found")

client = genai.Client(api_key=api_key)

# --------------------
# Load model + image
# --------------------
model = "gemini-2.5-flash"

image = Image.open("data/room.jpg")

# Resize image for faster inference
#image = image.resize((640, 480))
image = image.resize((320, 240))

# Compress image to reduce upload size
image.save("data/temp.jpg", format="JPEG", quality=40)

# Reload compressed image
image = Image.open("data/temp.jpg")
# --------------------
# Run inference
# --------------------
response = client.models.generate_content(
    model=model,
    contents=[
        "Briefly describe only the important navigational context, nearby obstacles, hazards, and major objects for a visually impaired person. Avoid unnecessary visual detail.",
        image
    ]
)

print(response.text)