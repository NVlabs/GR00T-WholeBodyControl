import matplotlib.pyplot as plt
import numpy as np


class ImageViewer:
    def __init__(self, title="Image Viewer", figsize=(8, 6), num_images=1, image_titles=None):
        self.title = title
        self.figsize = figsize
        self.num_images = num_images
        self.image_titles = image_titles or [f"Camera {i+1}" for i in range(num_images)]

        # Enable interactive mode before creating subplots
        plt.ion()

        if num_images == 1:
            self._fig, self._ax = plt.subplots(figsize=self.figsize)
            self._ax.set_title(self.title)
            self._im = self._ax.imshow(np.zeros((100, 100)))
            self._ax.axis("off")
            self._axes = [self._ax]
            self._images = [self._im]
        else:
            # Calculate grid dimensions
            cols = min(num_images, 3)  # Max 3 columns
            rows = (num_images + cols - 1) // cols

            # Adjust figure size based on number of images
            fig_width = self.figsize[0] * min(cols, 2)
            fig_height = self.figsize[1] * rows / 2

            self._fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))
            self._fig.suptitle(self.title)

            # Flatten axes array for easier access
            if num_images == 2:
                axes = [axes[0], axes[1]]
            elif rows == 1:
                axes = axes if cols > 1 else [axes]
            else:
                axes = axes.flatten()

            self._axes = []
            self._images = []

            for i in range(num_images):
                ax = axes[i]
                ax.set_title(self.image_titles[i])
                im = ax.imshow(np.zeros((100, 100)))
                ax.axis("off")
                self._axes.append(ax)
                self._images.append(im)

            # Hide unused subplots
            for i in range(num_images, len(axes)):
                axes[i].set_visible(False)

        # Show the figure initially to make window appear
        self._fig.show()

    def show(self, image_array):
        """Show a single image (backward compatibility)"""
        if self.num_images == 1:
            self._images[0].set_data(image_array)
        else:
            # If multiple viewers but single image provided, show in first viewer
            self._images[0].set_data(image_array)

        # non-blocking update
        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()

    def show_multiple(self, images):
        """Show multiple images"""
        for i, img in enumerate(images):
            if i < len(self._images) and img is not None:
                self._images[i].set_data(img)
                # Auto-adjust aspect ratio
                self._axes[i].set_aspect("auto")

        # non-blocking update
        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()

    def close(self):
        plt.close(self._fig)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
