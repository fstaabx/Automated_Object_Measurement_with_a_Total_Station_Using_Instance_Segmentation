# -----------------------------------------------------------------------------
# Interactive Hough Line Transform (Post-Canny Processing)
#
# This module provides an interactive GUI for the progressive probabilistic
# Hough Line Transform (PPHLT). It is the second script of the Computer Vision pipeline
# used in the automated total-station measurement workflow described in the
# associated publication.
#
# Context:
# - This script expects a Canny-edge image as input. No edge computation is done
#   here. The preceding segmentation + Canny step must provide a clean binary
#   edge map.
# - The segmentation step in the original workflow was based on a custom-trained
#   YOLOv8 model and a unique physical object. Without access to that model and
#   object, users cannot reproduce the same masks. Only the GUI logic presented
#   here is reusable.
#
# Functionality:
# - Adjustable PPHLT parameters: rho, theta, threshold, minLineLength, maxLineGap.
# - Live preview while parameters change.
# - Save function writing only the detected lines (no background) to disk.
# - Standalone usage supported, but originally designed to be called from an
#   external controller script.
# -----------------------------------------------------------------------------


import ttkbootstrap as ttk  # Import ttkbootstrap for a modern look
import tkinter as tk    # Import tkinter for GUI
from ttkbootstrap.constants import *    # Import constants from ttkbootstrap
from PIL import Image, ImageTk  # Import PIL for image handling
import cv2  # Import OpenCV for image processing
import numpy as np  # Import NumPy for numerical operations
import sys  # Import sys for system-specific parameters and functions


class HoughLineTransformApp:
    def __init__(self, second_gui, image, save_path=None):
        """
        Initialize the main application window and its components.
        Sets up the layout, loads the image, and prepares all controls.
        """
        self.second_gui = second_gui
        self.second_gui.title("Hough Line Transform - Interactive GUI")
        self.second_gui.geometry("1200x900")

        # Main frame to divide the left and right sections
        self.main_frame = ttk.Frame(self.second_gui)
        self.main_frame.pack(fill="both", expand=True)

        # Left frame for image display and parameter controls
        self.left_frame = ttk.Frame(self.main_frame, padding=20)
        self.left_frame.pack(side="left", fill="both", expand=True)

        # Right frame for code explanation
        self.right_frame = ttk.Frame(self.main_frame, padding=20, bootstyle="secondary")
        self.right_frame.pack(side="right", fill="both", expand=True)

        # Scrollable canvas for the left frame
        self.canvas = ttk.Canvas(self.left_frame)
        self.scrollbar = ttk.Scrollbar(self.left_frame, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)

        # Scrollable frame inside the canvas
        self.scrollable_frame = ttk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="n")
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        # Enable mouse wheel scrolling
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)

        # Load the input image
        self.image = image
        if self.image is None:
            raise ValueError("Image is None")

        # Check if the image is a grayscale image
        if len(self.image.shape) == 2:
            print("The image is already a grayscale image.")
        elif len(self.image.shape) == 3 and self.image.shape[2] == 1:
            print("The image is a grayscale image with 1 channel.")
        else:
            print("The image is a color image with 3 channels and has been converted to grayscale.")
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # If image is a string (path), extract image_name from it
        if isinstance(image, str):
            import os
            # image is expected to be like "edges_{image_name}.jpg"
            base = os.path.basename(image)
            if base.startswith("edges_") and base.endswith(".jpg"):
                self.image_name = base[len("edges_"):-len(".jpg")]
            else:
                self.image_name = os.path.splitext(base)[0]
            # Load the image from the path
            self.image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        else:
            # If image is already an array, set a default image_name
            self.image = image
            self.image_name = "unknown"

        # Set the save_path, or use a default if not provided
        if save_path is None:
            self.save_path = fr".\Hough_Line_Transform_Output_{self.image_name}.jpg"
        else:
            self.save_path = save_path

        # Create the UI components
        self.create_ui()
        self.create_right_panel()
        self.update_preview()

    def create_ui(self):
        """
        Create the user interface components for parameter adjustment and image preview.
        Includes sliders for all Hough parameters and buttons for reset/apply.
        """
        # Display area for the image
        self.image_label = ttk.Label(self.scrollable_frame, text="Image Preview", anchor="center")
        self.image_label.pack(pady=20)

        # Parameter: rho
        ttk.Label(self.scrollable_frame, text="Rho (Pixel Resolution):", font=("Helvetica", 10, "bold")).pack(anchor="center", pady=5)
        self.rho_var = ttk.IntVar(value=1)
        self.rho_label = ttk.Label(self.scrollable_frame, text=f"{self.rho_var.get()} px")
        self.rho_label.pack(anchor="center")
        rho_slider = ttk.Scale(
            self.scrollable_frame,
            from_=1,
            to=10,
            variable=self.rho_var,
            command=lambda _: self.update_parameter("rho"),
            orient="horizontal"
        )
        rho_slider.pack(anchor="center", pady=5)

        # Parameter: theta
        ttk.Label(self.scrollable_frame, text="Theta (Angle Resolution in Degrees):", font=("Helvetica", 10, "bold")).pack(anchor="center", pady=5)
        self.theta_var = ttk.IntVar(value=1)
        self.theta_label = ttk.Label(self.scrollable_frame, text=f"{self.theta_var.get()}°")
        self.theta_label.pack(anchor="center")
        theta_slider = ttk.Scale(
            self.scrollable_frame,
            from_=1,
            to=180,
            variable=self.theta_var,
            command=lambda _: self.update_parameter("theta"),
            orient="horizontal"
        )
        theta_slider.pack(anchor="center", pady=5)

        # Parameter: threshold
        ttk.Label(self.scrollable_frame, text="Threshold (Accumulator Hits):", font=("Helvetica", 10, "bold")).pack(anchor="center", pady=5)
        self.threshold_var = ttk.IntVar(value=15)
        self.threshold_label = ttk.Label(self.scrollable_frame, text=f"{self.threshold_var.get()}")
        self.threshold_label.pack(anchor="center")
        threshold_slider = ttk.Scale(
            self.scrollable_frame,
            from_=1,
            to=100,
            variable=self.threshold_var,
            command=lambda _: self.update_parameter("threshold"),
            orient="horizontal"
        )
        threshold_slider.pack(anchor="center", pady=5)

        # Parameter: minLineLength
        ttk.Label(self.scrollable_frame, text="Min Line Length (Pixels):", font=("Helvetica", 10, "bold")).pack(anchor="center", pady=5)
        self.min_line_length_var = ttk.IntVar(value=10)
        self.min_line_length_label = ttk.Label(self.scrollable_frame, text=f"{self.min_line_length_var.get()} px")
        self.min_line_length_label.pack(anchor="center")
        min_line_length_slider = ttk.Scale(
            self.scrollable_frame,
            from_=1,
            to=100,
            variable=self.min_line_length_var,
            command=lambda _: self.update_parameter("min_line_length"),
            orient="horizontal"
        )
        min_line_length_slider.pack(anchor="center", pady=5)

        # Parameter: maxLineGap
        ttk.Label(self.scrollable_frame, text="Max Line Gap (Pixels):", font=("Helvetica", 10, "bold")).pack(anchor="center", pady=5)
        self.max_line_gap_var = ttk.IntVar(value=30)
        self.max_line_gap_label = ttk.Label(self.scrollable_frame, text=f"{self.max_line_gap_var.get()} px")
        self.max_line_gap_label.pack(anchor="center")
        max_line_gap_slider = ttk.Scale(
            self.scrollable_frame,
            from_=1,
            to=100,
            variable=self.max_line_gap_var,
            command=lambda _: self.update_parameter("max_line_gap"),
            orient="horizontal"
        )
        max_line_gap_slider.pack(anchor="center", pady=5)

        reset_button = ttk.Button(self.scrollable_frame, text="Reset to Default", command=self.reset_to_default, bootstyle="warning")
        reset_button.pack(anchor="center", pady=10)

        apply_button = ttk.Button(self.scrollable_frame, text="Apply and Save", command=self.apply_and_save, bootstyle="success")
        apply_button.pack(anchor="center", pady=10)
        self.scrollable_frame.update_idletasks()
        # print("Children in scrollable_frame:", self.scrollable_frame.winfo_children())  # Debugging output

    def create_right_panel(self):
        """
        Create the right panel with explanations about the parameters and the process.
        This helps users understand the effect of each parameter.
        """
        # Set the background of the right panel to black
        self.right_frame.configure(style="Black.TFrame")

        # Title
        ttk.Label(self.right_frame, text="Code Explanation", font=("Helvetica", 14, "bold"), foreground="white").grid(row=0, column=0, columnspan=2, pady=10)

        # Description of the process
        ttk.Label(
            self.right_frame,
            text=(
                "This GUI performs Hough Line Transform to detect lines in an image. "
                "You can adjust parameters like Rho, Theta, Threshold, Min Line Length, "
                "and Max Line Gap to customize the line detection process. The result is "
                "visualized and can be saved for further use."
            ),
            wraplength=500,
            justify="left",
            foreground="white",
        ).grid(row=1, column=0, columnspan=2, pady=10)

        # Parameter description title
        ttk.Label(self.right_frame, text="Parameter Descriptions", font=("Helvetica", 14, "bold"), foreground="white").grid(row=2, column=0, columnspan=2, pady=10)

        # Parameter descriptions
        parameters = [
            (
                "Rho (Pixel Resolution)",
                (
                    "Defines the resolution of the accumulator in pixels.\n"
                    "- Smaller values allow detecting lines with finer precision.\n"
                    "- Larger values group nearby lines together, reducing precision."
                ),
            ),
            (
                "Theta (Angle Resolution)",
                (
                    "Defines the resolution of the accumulator in radians.\n"
                    "- Smaller values allow detecting lines at more specific angles.\n"
                    "- Larger values detect lines at broader angular ranges."
                ),
            ),
            (
                "Threshold",
                (
                    "Defines the minimum number of intersections required to detect a line.\n"
                    "- Higher values focus on prominent, well-defined lines.\n"
                    "- Lower values include weaker or less distinct lines."
                ),
            ),
            (
                "Min Line Length",
                (
                    "Defines the minimum length of a line in pixels.\n"
                    "- Smaller values detect shorter lines, which may include noise.\n"
                    "- Larger values filter out shorter lines, focusing on longer ones."
                ),
            ),
            (
                "Max Line Gap",
                (
                    "Defines the maximum gap between line segments to connect them.\n"
                    "- Smaller values keep line segments separate unless they are very close.\n"
                    "- Larger values connect distant segments into a single line."
                ),
            ),
            (
                "More Information",
                (
                    "For more details on the Hough Line Transform, refer to the OpenCV documentation:\n"
                    "https://docs.opencv.org/3.4/d9/db0/tutorial_hough_lines.html"
                )
            )
        ]

        for i, (title, description) in enumerate(parameters, start=3):
            ttk.Label(self.right_frame, text=f"{i - 2}. {title}", font=("Helvetica", 12, "bold"), foreground="white").grid(row=i, column=0, sticky="w", padx=10, pady=5)
            ttk.Label(
                self.right_frame,
                text=description,
                wraplength=250,
                justify="left",
                foreground="white",
            ).grid(row=i, column=1, sticky="w", padx=10, pady=5)

    def update_parameter(self, parameter):
        """
        Update the parameter labels and refresh the preview when a parameter changes.
        """
        if parameter == "rho":
            self.rho_label.config(text=f"{self.rho_var.get()} px")
        elif parameter == "theta":
            self.theta_label.config(text=f"{self.theta_var.get()}°")
        elif parameter == "threshold":
            self.threshold_label.config(text=f"{self.threshold_var.get()}")
        elif parameter == "min_line_length":
            self.min_line_length_label.config(text=f"{self.min_line_length_var.get()} px")
        elif parameter == "max_line_gap":
            self.max_line_gap_label.config(text=f"{self.max_line_gap_var.get()} px")
        
        # Refresh the preview
        self.update_preview()

    def _on_mousewheel(self, event):
        """
        Enable mouse wheel scrolling for the canvas.
        """
        self.canvas.yview_scroll(-1 * (event.delta // 120), "units")

    def apply_and_save(self):
        """
        Apply the Hough Line Transform and save the result to a file.
        The result image contains only the detected lines on a black background.
        """
        lines = cv2.HoughLinesP(
            self.image,
            rho=self.rho_var.get(),
            theta=self.theta_var.get() * (np.pi / 180),
            threshold=self.threshold_var.get(),
            minLineLength=self.min_line_length_var.get(),
            maxLineGap=self.max_line_gap_var.get()
        )

        # Debugging: Check if lines are detected
        if lines is not None:
            print(f"Detected lines: {len(lines)}")
        else:
            print("No lines detected.")

        # Create a completely black image for the result (same size as input, 3 channels)
        black_result = np.zeros((self.image.shape[0], self.image.shape[1], 3), dtype=np.uint8)

        # Draw the lines on the black image
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # Debugging: Check the coordinates
                if 0 <= x1 < self.image.shape[1] and 0 <= y1 < self.image.shape[0] and \
                0 <= x2 < self.image.shape[1] and 0 <= y2 < self.image.shape[0]:
                    print(f"Line coordinates within bounds: ({x1}, {y1}) -> ({x2}, {y2})") # Debugging output
                else:
                    print(f"Line coordinates out of bounds: ({x1}, {y1}) -> ({x2}, {y2})") # Debugging output
                # Draw the line (red)
                cv2.line(black_result, (x1, y1), (x2, y2), (0, 255, 0), 1)  # Green lines with thickness 2

        self.result_image = black_result  # Set the result image to the black image with lines

        # Save the result image to the specified path
        if cv2.imwrite(self.save_path, self.result_image):
            print(f"Result image successfully saved to {self.save_path}.")
        else:
            print("Error saving the result image.")

        # Close the application
        self.second_gui.quit()
        self.second_gui.destroy()

    def reset_to_default(self):
        """
        Reset all parameters to their default values and refresh the preview.
        """
        self.rho_var.set(1)
        self.theta_var.set(1)
        self.threshold_var.set(15)
        self.min_line_length_var.set(10)
        self.max_line_gap_var.set(30)
        self.update_preview()

    def update_preview(self):
        """
        Update the preview image in the GUI based on the current parameters.
        Detected lines are drawn in green for visualization.
        """
        # Create a copy of the original image
        preview_image = cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)  # Convert grayscale to BGR

        # Perform the Hough Line Transform
        lines = cv2.HoughLinesP(
            self.image,
            rho=self.rho_var.get(),
            theta=self.theta_var.get() * (np.pi / 180),
            threshold=self.threshold_var.get(),
            minLineLength=self.min_line_length_var.get(),
            maxLineGap=self.max_line_gap_var.get()
        )

        # Draw the lines on the preview image
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(preview_image, (x1, y1), (x2, y2), (0, 255, 0), 1)  # Green lines with thickness 2

        # Convert the image for display in the GUI
        preview_image = cv2.cvtColor(preview_image, cv2.COLOR_BGR2RGB)
        preview_image = Image.fromarray(preview_image)

        # Scale the image proportionally to fit the window
        max_width, max_height = 800, 600
        orig_width, orig_height = preview_image.size
        scale_w = max_width / orig_width
        scale_h = max_height / orig_height
        scale = min(scale_w, scale_h)
        new_width = int(orig_width * scale)
        new_height = int(orig_height * scale)
        preview_image = preview_image.resize((new_width, new_height), Image.LANCZOS)

        # Save the image as an attribute
        self.photo = ImageTk.PhotoImage(preview_image)

        # Update the image label
        if self.image_label.winfo_exists():
            self.image_label.config(image=self.photo)
        else:
            print("The window or the image label component no longer exists.")

def run_Hough_from_another_script(image, root, save_path):
    """
    Run the Hough Line Transform application as a Toplevel window.
    Returns the result image with detected lines.
    """

    # Check if the image is valid
    if image is None:
        print("Error: The provided image is None. Check the image path or source.")
        return None

    # Create a second GUI as a Toplevel window
    second_gui = tk.Toplevel(root)

    # Ensure program exits when window is closed
    def on_close():
        second_gui.destroy()
        sys.exit(0)
    second_gui.protocol("WM_DELETE_WINDOW", on_close)

    app = HoughLineTransformApp(second_gui, image, save_path=save_path)

    # Start the event loop for the second GUI
    second_gui.mainloop()

    # Return the result image
    return app.result_image


def main():
    # -----------------------------------------------------------------------------
    # Standalone usage example:
    # Loads an image, opens the interactive Hough Line Transform GUI,
    # and saves the result image with detected lines.
    # -----------------------------------------------------------------------------
    # Define the input image path
    image_path = r".\edges_4_0073.jpg"

    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Check if the image is a grayscale image
    if len(image.shape) == 2:
        print("The image is already a grayscale image.")
    elif len(image.shape) == 3 and image.shape[2] == 1:
        print("The image is a grayscale image with 1 channel.")
    else:
        print("The image is a color image with 3 channels.")

    # Create the main Tk instance and apply darkly style
    root = tk.Tk()
    root.withdraw()
    style = ttk.Style("darkly")

    # Define the save path for the result image
    save_path = r".\Hough_Line_Transform.jpg"

    # Start the Hough function
    result_image = run_Hough_from_another_script(image, root, save_path)



if __name__ == "__main__":

    main()
