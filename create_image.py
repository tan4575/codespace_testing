from PIL import Image, ImageDraw

# Your colors
colors = [
    "#E53935",
    "#FB8C00",
    "#FDD835",
    "#43A047",
    "#00ACC1",
    "#1E88E5",
    "#8E24AA",
    "#E91E63",
    "#FAFAFA",
    "#212121",
    "#757575",
    "#6D4C41"
]

# Image settings
block_width = 320
height = 320

for i in range(len(colors)):
    # Create image
    img = Image.new("RGB", (block_width, height))

    draw = ImageDraw.Draw(img)

    # # Draw color blocks
    draw.rectangle([0, 0, block_width, height], fill=colors[i])

    # Save image
    img.save(f"./data/color_palette_{i}.png")
