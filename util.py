from PIL import Image
import matplotlib.pyplot as plt
from io import BytesIO

def combine_plots(figures, output_path, orientation='horizontal'):
    # Convert the matplotlib plots to PIL images
    images = [plotnine_to_image(fig) for fig in figures]

    # Get the dimensions of the images
    widths, heights = zip(*(img.size for img in images))

    if orientation == 'horizontal':
        # Calculate combined width and maximum height
        combined_width = sum(widths)
        combined_height = max(heights)
        combined_img = Image.new("RGB", (combined_width, combined_height))

        # Paste the images side by side
        x_offset = 0
        for img in images:
            combined_img.paste(img, (x_offset, 0))
            x_offset += img.width
    elif orientation == 'vertical':
        # Calculate maximum width and combined height
        combined_width = max(widths)
        combined_height = sum(heights)
        combined_img = Image.new("RGB", (combined_width, combined_height))

        # Paste the images one below the other
        y_offset = 0
        for img in images:
            combined_img.paste(img, (0, y_offset))
            y_offset += img.height
    else:
        raise ValueError("Orientation must be 'horizontal' or 'vertical'")

    # Save the combined image
    combined_img.save(output_path)

def plotnine_to_image(plot):
    """Convert a Plotnine figure to a PIL Image."""
    buf = BytesIO()
    plot.save(buf, format='png', verbose=False)
    buf.seek(0)
    return Image.open(buf)