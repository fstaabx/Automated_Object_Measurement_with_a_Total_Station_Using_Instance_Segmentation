# -----------------------------------------------------------------------------
# Harris Corner Detection and Clustering Interactive GUI Application
#
# This script provides an interactive GUI for performing Harris Corner Detection
# and clustering the detected corners using DBSCAN. A Canny edge detection and Hough Transform
# should be applied before running this script to prepare the image and enhance corner detection.
# The user can:
# - Adjust parameters for Gaussian Blur, Harris Corner Detection, and DBSCAN clustering.
# - Instantly preview the effect of parameter changes on the detected corners.
# - See the number of detected points in real time.
# - Save the result image with marked corners and a text file with the corner coordinates.
# - View explanations for each parameter and the overall process in a dedicated panel.
#
# The GUI is implemented with Tkinter and ttkbootstrap for a modern look.
# The script can be called from another script or run standalone.
# -----------------------------------------------------------------------------


import ttkbootstrap as ttk  # Import ttkbootstrap for modern GUI styling
import tkinter as tk    # Import tkinter for GUI components
from PIL import Image, ImageTk  # Import PIL for image handling in Tkinter
import cv2  # Import OpenCV for image processing
import numpy as np  # Import NumPy for numerical operations
from sklearn.cluster import DBSCAN  # Import DBSCAN for clustering detected corners
import sys  # Import sys for system-level operations


class HarrisCornerDetectionApp:
    def __init__(self, third_gui, image_path, img_output_path, txt_output_path):
        """
        Initialize the main application window and layout.
        Sets up the GUI, loads the image, and prepares all controls.
        """

        # Initialize the main GUI window
        self.third_gui = third_gui
        self.third_gui.title("Harris Corner Detection - Interactive GUI")
        self.third_gui.geometry("1200x900")

        # Main frame to divide the left and right sections
        self.main_frame = ttk.Frame(self.third_gui)
        self.main_frame.pack(fill="both", expand=True)

        # Left frame for image display and parameter controls
        self.left_frame = ttk.Frame(self.main_frame, padding=20)
        self.left_frame.pack(side="left", fill="both", expand=True)

        # Right frame for additional information and descriptions
        self.right_frame = ttk.Frame(self.main_frame, padding=20, bootstyle="secondary")
        self.right_frame.pack(side="right", fill="both", expand=True)

        # Canvas for scrollable content in the left frame
        self.canvas = ttk.Canvas(self.left_frame)
        self.scrollable_frame = ttk.Frame(self.canvas)

        # Bind the scrollable frame to the canvas
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="n")

        # Pack the canvas
        self.canvas.pack(side="left", fill="both", expand=True)

        # Enable mouse wheel scrolling
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)

        # Central frame inside the scrollable area for widgets
        self.central_frame = ttk.Frame(self.scrollable_frame, padding=20)
        self.central_frame.pack(anchor="center")

        # Default paths for input and output images
        self.default_image_path = image_path
        self.output_image_path = img_output_path
        self.output_txt_path = txt_output_path

        # Variables to store images and processing results
        self.image = None
        self.processed_image = None
        self.corners = None
        self.merged_corners = None

        # Create the UI components
        self.create_ui()
        self.create_right_panel()
        self.load_image(self.default_image_path)

    def create_ui(self):
        """
        Create all user interface components for parameter adjustment and image preview.
        Includes dropdowns for all parameters and a live preview area.
        """

        # Define a custom style for the image label
        style = ttk.Style()
        style.configure("Flat.TLabel", relief="flat", borderwidth=0)

        # Display area for the image
        self.image_label = ttk.Label(self.central_frame, text="Image Preview", anchor="center", style="Flat.TLabel")
        self.image_label.pack(pady=20)

        # Gaussian Blur parameter
        ttk.Label(self.central_frame, text="Gaussian Blur Kernel Size:", font=("Helvetica", 10, "bold")).pack(anchor="center", pady=5)
        self.blur_kernel_var = ttk.StringVar(value="5")
        blur_kernel_menu = ttk.OptionMenu(self.central_frame, self.blur_kernel_var, "5", "3", "5", "7", "9", "11", "13", "15", "17", command=self.update_preview)
        blur_kernel_menu.pack(anchor="center", pady=5)

        # Harris Corner Detection parameters (blockSize)
        ttk.Label(self.central_frame, text="Neighborhood Size (blockSize):", font=("Helvetica", 10, "bold")).pack(anchor="center", pady=5)
        self.block_size_var = ttk.StringVar(value="3")
        block_size_menu = ttk.OptionMenu(self.central_frame, self.block_size_var, "3", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", command=self.update_preview)
        block_size_menu.pack(anchor="center", pady=5)

        # Sobel Aperture Size (ksize)
        ttk.Label(self.central_frame, text="Sobel Aperture Size (ksize):", font=("Helvetica", 10, "bold")).pack(anchor="center", pady=5)
        self.ksize_var = ttk.StringVar(value="3")
        ksize_menu = ttk.OptionMenu(self.central_frame, self.ksize_var, "3", "3", "5", "7", "9", "11", "13", "15", "17", "19", command=self.update_preview)
        ksize_menu.pack(anchor="center", pady=5)

        # Harris Sensitivity (k)
        ttk.Label(self.central_frame, text="Harris Sensitivity (k):", font=("Helvetica", 10, "bold")).pack(anchor="center", pady=5)
        self.k_var = ttk.StringVar(value="0.04")
        k_menu = ttk.OptionMenu(self.central_frame, self.k_var, "0.04", "0.02", "0.04", "0.06", "0.08", "0.1", "0.12", "0.14", "0.16",  command=self.update_preview)
        k_menu.pack(anchor="center", pady=5)

        # Threshold for Harris response
        ttk.Label(self.central_frame, text="Corner Response Threshold:", font=("Helvetica", 10, "bold")).pack(anchor="center", pady=5)
        self.harris_thresh_var = ttk.DoubleVar(value=0.01)
        thresh_frame = ttk.Frame(self.central_frame)
        thresh_frame.pack(anchor="center", pady=5)

        thresh_slider = ttk.Scale(
            thresh_frame,
            from_=0.01,
            to=10,
            variable=self.harris_thresh_var,
            command=lambda _: self.update_preview(),
            orient="horizontal",
            length=200
        )
        thresh_slider.pack(side="left")

        self.thresh_value_label = ttk.Label(thresh_frame, text=f"{self.harris_thresh_var.get():.2f}")
        self.thresh_value_label.pack(side="left", padx=10)

        # Update the label when the slider moves
        def update_thresh_label(*args):
            self.thresh_value_label.config(text=f"{self.harris_thresh_var.get():.2f}")
        self.harris_thresh_var.trace_add("write", update_thresh_label)

        # Number of Points
        ttk.Label(self.central_frame, text="Number of Points", font=("Helvetica", 10, "bold")).pack(anchor="center", pady=(10, 2))
        self.num_points_var = ttk.StringVar(value="0")
        ttk.Label(self.central_frame, textvariable=self.num_points_var, font=("Helvetica", 10)).pack(anchor="center", pady=(0, 10))

        # DBSCAN parameters (eps and min_samples)
        ttk.Label(self.central_frame, text="DBSCAN - Max Distance (eps):", font=("Helvetica", 10, "bold")).pack(anchor="center", pady=5)
        self.eps_var = ttk.StringVar(value="10")
        eps_menu = ttk.OptionMenu(self.central_frame, self.eps_var, "10", "5", "10", "15", "20", "25", "30", "35", "40", command=self.update_preview)
        eps_menu.pack(anchor="center", pady=5)

        ttk.Label(self.central_frame, text="DBSCAN - Min Points (min_samples):", font=("Helvetica", 10, "bold")).pack(anchor="center", pady=5)
        self.min_samples_var = ttk.StringVar(value="1")
        min_samples_menu = ttk.OptionMenu(self.central_frame, self.min_samples_var, "1", "1", "2", "3", "4", "5", "6", "7", "8", command=self.update_preview)
        min_samples_menu.pack(anchor="center", pady=5)

        # Button to apply parameters and save the result
        ttk.Button(self.central_frame, text="Apply Parameters", command=self.apply_and_save, bootstyle="success").pack(pady=20)
        # self.update_preview()

    def create_right_panel(self):
        """
        Create the right panel with explanations about the parameters and the process.
        Helps users understand the effect of each parameter.
        """

        # Set the background of the right panel to black
        self.right_frame.configure(style="Black.TFrame")

        # Title for the right panel
        ttk.Label(self.right_frame, text="Process Description", font=("Helvetica", 14, "bold"), foreground="white").grid(row=0, column=0, columnspan=2, pady=10)

        # Description of the process
        ttk.Label(
            self.right_frame,
            text=(
                "This GUI performs Harris Corner Detection followed by clustering "
                "of the detected corners using DBSCAN. The user can adjust various "
                "parameters for image processing and clustering. The processed result "
                "is visualized and saved."
            ),
            wraplength=500,
            justify="left",
            foreground="white",
        ).grid(row=1, column=0, columnspan=2, pady=10)

        # Parameter description title
        ttk.Label(self.right_frame, text="Parameter Descriptions", font=("Helvetica", 14, "bold"), foreground="white").grid(row=2, column=0, columnspan=2, pady=10)

        # Parameter descriptions 
        # Parameter 1: Gaussian Blur Kernel Size
        ttk.Label(self.right_frame, text="1. Gaussian Blur Kernel Size", font=("Helvetica", 12, "bold"), foreground="white").grid(row=3, column=0, sticky="w", padx=10, pady=5)
        ttk.Label(
            self.right_frame,
            text=(
                "Applies Gaussian smoothing to reduce image noise before corner detection.\n"
                "Values: 3, 5, 7, 9, 11, 13, 15, 17\n"
                "- Small: Retains fine details.\n"
                "- Large: Stronger noise suppression."
            ),
            wraplength=250,
            justify="left",
            foreground="white",
        ).grid(row=3, column=1, sticky="w", padx=10, pady=5)

        # Parameter 2: Neighborhood Size
        ttk.Label(self.right_frame, text="2. Neighborhood Size (blockSize)", font=("Helvetica", 12, "bold"), foreground="white").grid(row=4, column=0, sticky="w", padx=10, pady=5)
        ttk.Label(
            self.right_frame,
            text=(
                "Size of the window considered for computing the gradient covariance matrix.\n"
                "Values: 2–11\n"
                "- Small: Detects finer corners.\n"
                "- Large: Captures broader corner structures."
            ),
            wraplength=250,
            justify="left",
            foreground="white",
        ).grid(row=4, column=1, sticky="w", padx=10, pady=5)

        # Parameter 3: Sobel Aperture Size
        ttk.Label(self.right_frame, text="3. Sobel Aperture Size (ksize)", font=("Helvetica", 12, "bold"), foreground="white").grid(row=5, column=0, sticky="w", padx=10, pady=5)
        ttk.Label(
            self.right_frame,
            text=(
                "Kernel size used in the Sobel operator for gradient calculation.\n"
                "Values: 3, 5, 7, 9, 11, 13, 15, 17, 19\n"
                "- Small: Captures sharper edges.\n"
                "- Large: Smoother gradients, more robust to noise."
            ),
            wraplength=250,
            justify="left",
            foreground="white",
        ).grid(row=5, column=1, sticky="w", padx=10, pady=5)

        # Parameter 4: Harris Sensitivity
        ttk.Label(self.right_frame, text="4. Harris Sensitivity (k)", font=("Helvetica", 12, "bold"), foreground="white").grid(row=6, column=0, sticky="w", padx=10, pady=5)
        ttk.Label(
            self.right_frame,
            text=(
                "Harris detector free parameter controlling sensitivity to corners.\n"
                "Values: 0.02 – 0.16\n"
                "- Small: Detects many corners, including weak ones.\n"
                "- Large: Emphasizes strong corners, filters weak ones."
            ),
            wraplength=250,
            justify="left",
            foreground="white",
        ).grid(row=6, column=1, sticky="w", padx=10, pady=5)

        # Parameter 5: Corner Response Threshold
        # Uncommented for better display on the laptop screen
        # ttk.Label(self.right_frame, text="5. Corner Response Threshold", font=("Helvetica", 12, "bold"), foreground="white").grid(row=6, column=0, sticky="w", padx=10, pady=5)
        # ttk.Label(
        #     self.right_frame,
        #     text=(
        #         "Sets the minimum relative strength required for a pixel to be considered as a corner.\n"
        #         "This value is a percentage of the strongest detected corner response in the image.\n"
        #         "Values: 0.01 – 10\n"
        #         "- Lower values: More points are detected, including weak or noisy corners.\n"
        #         "- Higher values: Only the strongest, most prominent corners are kept."
        #     ),
        #     wraplength=250,
        #     justify="left",
        #     foreground="white",
        # ).grid(row=6, column=1, sticky="w", padx=10, pady=5)

        # Parameter 5.1: DBSCAN Max Distance
        ttk.Label(self.right_frame, text="5. DBSCAN - Max Distance (eps)", font=("Helvetica", 12, "bold"), foreground="white").grid(row=7, column=0, sticky="w", padx=10, pady=5)
        ttk.Label(
            self.right_frame,
            text=(
                "Defines the maximum distance between two points to be considered neighbors.\n"
                "Values: 5, 10, 15, 20, 25, 30, 35, 40\n"
                "- Small: Detects fine-grained clusters.\n"
                "- Large: Merges nearby points into broader clusters."
            ),
            wraplength=250,
            justify="left",
            foreground="white",
        ).grid(row=7, column=1, sticky="w", padx=10, pady=5)

        # Parameter 6: DBSCAN Min Points
        ttk.Label(self.right_frame, text="6. DBSCAN - Min Points (min_samples)", font=("Helvetica", 12, "bold"), foreground="white").grid(row=8, column=0, sticky="w", padx=10, pady=5)
        ttk.Label(
            self.right_frame,
            text=(
                "Defines the number of points (including the point itself) required to form a dense region.\n"
                "Values: 1, 2, 3, 4, 5, 6, 7, 8\n"
                "- Small: More sensitive, includes noise.\n"
                "- Large: Requires denser clusters, filters out noise."
            ),
            wraplength=250,
            justify="left",
            foreground="white",
        ).grid(row=8, column=1, sticky="w", padx=10, pady=5)

    def load_image(self, path):
        """
        Load the image in grayscale and update the preview.
        Raises FileNotFoundError if the image cannot be loaded.
        """

        # Load the image in grayscale
        self.image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if self.image is None:
            raise FileNotFoundError(f"Image not found at '{path}'")
        self.update_preview()

    def update_preview(self, *args):
        """
        Update the preview image and number of points based on current parameters.
        Applies Gaussian Blur, Harris Corner Detection, and DBSCAN clustering.
        """

        # Extract parameters from the UI
        kernel_size = int(self.blur_kernel_var.get()) | 1
        block_size = int(self.block_size_var.get())
        ksize = int(self.ksize_var.get()) | 1
        k = float(self.k_var.get())
        eps = float(self.eps_var.get())
        min_samples = int(self.min_samples_var.get())

        # Apply Gaussian Blur
        processed_image = cv2.GaussianBlur(self.image, (kernel_size, kernel_size), 0)

        # Perform Harris Corner Detection
        corners = cv2.cornerHarris(processed_image, blockSize=block_size, ksize=ksize, k=k)
        corners = cv2.dilate(corners, None)
        thresh_percent = float(self.harris_thresh_var.get())
        threshold = (thresh_percent / 100.0) * corners.max()
        corner_points = np.argwhere(corners > threshold)

        # Perform DBSCAN clustering
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(corner_points)
        labels = db.labels_
        merged_corners = []
        for label in set(labels):
            if label == -1:
                continue
            group = corner_points[labels == label]
            mean_x = int(np.mean(group[:, 0]))
            mean_y = int(np.mean(group[:, 1]))
            merged_corners.append((mean_x, mean_y))

        # Visualize the result
        vis_image = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2BGR)
        for x, y in merged_corners:
            cv2.circle(vis_image, (y, x), 5, (255, 0, 0), -1)
        self.display_image(vis_image)

        # Update the number of points detected
        self.num_points_var.set(str(len(merged_corners)))


    def apply_and_save(self):
        """
        Apply all processing steps and save the result image and corner coordinates.
        The result image is saved as an image file, and the corner coordinates as a text file.
        """

        # Extract parameters and process the image
        kernel_size = int(self.blur_kernel_var.get()) | 1
        block_size = int(self.block_size_var.get())
        ksize = int(self.ksize_var.get()) | 1
        k = float(self.k_var.get())
        eps = float(self.eps_var.get())
        min_samples = int(self.min_samples_var.get())

        # Apply Gaussian Blur
        processed_image = cv2.GaussianBlur(self.image, (kernel_size, kernel_size), 0)
        # Explanation:
        # `cv2.GaussianBlur`: Applies a Gaussian blur to the input image to reduce noise and smooth the image. Also double edges shoudl be better handled.
        # - Parameters:
        # `self.image`: The input image (grayscale) to be smoothed.
        # `(kernel_size, kernel_size)`: The size of the Gaussian kernel (must be odd, e.g., 3, 5, 7, etc.).
        # A larger kernel size results in stronger smoothing, which can help reduce noise but may blur fine details.
        # A smaller kernel size retains more details but may not effectively reduce noise.
        # `0`: The standard deviation of the Gaussian kernel is calculated automatically based on the kernel size.
        # Purpose:
        # Reduces noise in the image, which is important for improving the accuracy of subsequent corner detection.
        # Smooths the image to make corner detection more robust to small variations in pixel intensity.
        # Result:
        # `processed_image`: The smoothed version of the input image, ready for Harris Corner Detection.

        # This step detects corners in the processed image using the Harris Corner Detection algorithm.
        corners = cv2.cornerHarris(processed_image, blockSize=block_size, ksize=ksize, k=k)
        # Explanation:
        # `processed_image`: The input image, which has been smoothed using Gaussian Blur.
        # `blockSize`: The size of the neighborhood considered for corner detection.
        # A larger block size considers a bigger neighborhood, which can detect larger structures.
        # `ksize`: The aperture size of the Sobel operator used to calculate image gradients.
        # A larger ksize makes the algorithm more robust to noise but less sensitive to small details.
        # `k`: The Harris detector free parameter, typically between 0.04 and 0.06.
        # A smaller value makes the algorithm more sensitive to corners.
        # Result:
        # `corners`: A grayscale image where each pixel value represents the likelihood of being a corner.
        # Higher values indicate a higher likelihood of being a corner.

        # Dilate the corners to enhance them
        corners = cv2.dilate(corners, None)
        # Explanation:
        # `cv2.dilate`: This operation expands the bright regions in the `corners` image.
        # Purpose:
        # Enhances the detected corners by making them more prominent.
        # Helps to merge nearby corner responses into a single region.
        # `None`: No specific kernel is provided, so OpenCV uses a default 3x3 rectangular kernel.

        # Calculate a threshold to filter out weak corner responses
        thresh_percent = float(self.harris_thresh_var.get())
        threshold = (thresh_percent / 100.0) * corners.max()
        # Explanation:
        # `self.harris_thresh_var.get()`: Retrieves the threshold percentage set by the user.
        # `corners.max()`: Gets the maximum value in the `corners` image.
        # The threshold is calculated as a percentage of the maximum corner response value.

        # Extract the coordinates of the corners that exceed the threshold
        corner_points = np.argwhere(corners > threshold)
        # Explanation:
        # `corners > threshold`: Creates a binary mask where pixels with values greater than the threshold are marked as `True`.
        # `np.argwhere`: Returns the coordinates (row, column) of all `True` pixels in the binary mask.
        # Result:
        # `corner_points`: A list of coordinates representing the detected corners in the image.
        # Each coordinate is in the format `[row, column]`, corresponding to the pixel's position in the image.)

        # Perform DBSCAN clustering
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(corner_points)
        # Explanation:
        # `DBSCAN`: Density-Based Spatial Clustering of Applications with Noise.
        # It groups points that are closely packed together while marking points in low-density regions as noise.
        # Parameters:
        # `eps`: The maximum distance between two points to be considered as part of the same cluster.
        # A smaller `eps` results in smaller, tighter clusters.
        # `min_samples`: The minimum number of points required to form a cluster.
        # A larger `min_samples` results in fewer, larger clusters.
        # `corner_points`: The list of coordinates of detected corners from the Harris Corner Detection step.
        # Result:
        # `db`: A DBSCAN object that contains the clustering results.

        labels = db.labels_  # Cluster labels for each corner point
        merged_corners = []

        for label in set(labels):
            if label == -1:  # Skip noise points
                continue
            group = corner_points[labels == label]  # Points in the current cluster
            mean_x = int(np.mean(group[:, 0]))  # Average row (y-coordinate)
            mean_y = int(np.mean(group[:, 1]))  # Average column (x-coordinate)
            merged_corners.append((mean_x, mean_y))  # Add averaged cluster center

        # Visualize and save the result
        vis_image = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2BGR)
        for x, y in merged_corners:
            cv2.circle(vis_image, (y, x), 5, (255, 0, 0), -1)

        success = cv2.imwrite(self.output_image_path, vis_image)
        if success:
            print(f"Image successfully saved at: {self.output_image_path}")
        else:
            print(f"Error saving the image at: {self.output_image_path}")

        # Save the corner points in a tuple form to a text file
        with open(self.output_txt_path, "w") as f:
            for x, y in merged_corners:
                f.write(f"({x}, {y})\n")
        print(f"Corner points saved at: {self.output_txt_path}")

        # Close the GUI
        self.third_gui.quit()
        self.third_gui.destroy()

    def display_image(self, image):
        """
        Display the (optionally color) image, scaled proportionally for the GUI.
        Converts grayscale to RGB if needed and resizes to fit the preview area.
        """

        # If the image is grayscale, convert it to RGB for display
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # Window size limits
        max_width = 800
        max_height = 600

        # Get original image dimensions
        orig_height, orig_width = image.shape[:2]

        # Calculate scaling factors
        scale_w = max_width / orig_width
        scale_h = max_height / orig_height
        scale = min(scale_w, scale_h)

        # Calculate new dimensions
        new_width = int(orig_width * scale)
        new_height = int(orig_height * scale)

        # Resize the image to fit within the limits while maintaining aspect ratio
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)

        # Convert the image to a format suitable for Tkinter
        image = Image.fromarray(image)
        photo = ImageTk.PhotoImage(image)
        self.image_label.config(image=photo)
        self.image_label.image = photo

    def _on_mousewheel(self, event):
        """
        Enable mouse wheel scrolling for the canvas.
        """

        # Enable mouse wheel scrolling
        self.canvas.yview_scroll(-1 * (event.delta // 120), "units")

def run_harris_gui_from_another_script(image_path, img_output_path, txt_output_path, root):
    """
    Run the Harris Corner Detection GUI as a Toplevel window and return output paths.
    """    

    # Create a third GUI as a Toplevel window
    third_gui = tk.Toplevel(root)

    # Ensure program exits when window is closed
    def on_close():
        third_gui.destroy()
        sys.exit(0)
    third_gui.protocol("WM_DELETE_WINDOW", on_close)

    app = HarrisCornerDetectionApp(third_gui, image_path, img_output_path, txt_output_path)

    # Start the event loop for the second GUI
    third_gui.mainloop()

    # Return the result image
    return app.output_image_path, app.output_txt_path


def main():
    """
    Standalone usage example: launches the GUI and prints output file paths after processing.
    """
        
    # Create the main Tk instance
    root = tk.Tk()
    root.withdraw()
    style = ttk.Style("darkly")

    # Define paths for input and output images
    image_path = r".\Hough_Line_Transform.jpg"
    img_output_path = r".\Harris_corner_detction.jpg"
    txt_output_path = r".\Harris_corner_detction.txt"

    # Call the Harris Corner Detection GUI
    output_image, output_txt = run_harris_gui_from_another_script(image_path, img_output_path, txt_output_path, root)

    print(f"Output image saved at: {output_image}")
    print(f"Output text file saved at: {output_txt}")

if __name__ == "__main__":
    main()