"""
Render the three Paradiso XXX scenes in your trained Persian-miniature style.
The story stays Dante; the picture turns into a Safavid / Shahnameh manuscript page.

    python 02_generate_paradiso.py
"""
import glob, torch
from diffusers import FluxPipeline

LORA_DIR   = "output/flux_persian_miniature"
LORA_SCALE = 0.85
SEED       = 42

# Canto XXX, beat by beat, in Persian visual grammar
SCENES = {
    "01_river_of_light":
        "shahnameh persian miniature style. Two pilgrims — a poet and his radiant guide — "
        "stand at the edge of a luminous river of golden light flowing between two banks of "
        "springtime flowers. Living sparks rise from the river like rubies set in gold and "
        "settle among the blossoms. Safavid manuscript page, gold-speckled ground, lapis-blue "
        "sky, turquoise water, delicate stylized trees, ornamental floral border, flat symbolic "
        "space, fine linework, divine radiance.",

    "02_river_becomes_rose":
        "shahnameh persian miniature style. The straight river of golden light curves and closes "
        "into a vast circle; its stream transforms into a great rose of light, tiered petals of "
        "glowing radiance opening across the page, as a poet and his luminous guide gaze upward. "
        "Safavid illuminated manuscript, gold ground, lapis and turquoise, concentric rings of "
        "light, ornamental border, flat sacred composition, intricate detail.",

    "03_celestial_rose":
        "shahnameh persian miniature style. An immense rose of white light: tier upon tier of "
        "blessed souls in a circular amphitheater of radiance, haloed figures in elegant robes, "
        "winged messengers moving among them carrying peace, a single point of pure light at the "
        "center. A small poet stands below beside his guide. Safavid manuscript page, gold-speckled "
        "heavens, lapis and turquoise, ranked figures, ornamental floral framing, flat symbolic "
        "perspective, fine miniature linework, overwhelming divine luminosity.",
}

print("loading Flux…")
pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)

ckpts = sorted(glob.glob(f"{LORA_DIR}/*.safetensors"))
assert ckpts, f"no .safetensors found in {LORA_DIR} — train first"
pipe.load_lora_weights(ckpts[-1])     # newest checkpoint
pipe.to("cuda")

for name, prompt in SCENES.items():
    print("rendering", name)
    img = pipe(
        prompt,
        height=1024, width=768,        # tall, like a manuscript folio
        num_inference_steps=28,
        guidance_scale=3.5,
        joint_attention_kwargs={"scale": LORA_SCALE},
        generator=torch.Generator("cuda").manual_seed(SEED),
    ).images[0]
    img.save(f"{name}.png")

print("done -> 01_river_of_light.png, 02_river_becomes_rose.png, 03_celestial_rose.png")
