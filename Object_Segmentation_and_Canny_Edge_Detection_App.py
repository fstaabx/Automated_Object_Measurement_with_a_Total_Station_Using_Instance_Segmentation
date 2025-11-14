# -----------------------------------------------------------------------------
# YOLOv8 Segmentation + Interactive Canny Edge Detection (GUI)
#
# This script is the computer-vision preprocessing component used in the original
# automated total-station measurement workflow. It runs YOLOv8-based instance
# segmentation on images from an input folder, extracts the detected object mask,
# crops the object region, and launches an interactive Canny GUI to determine
# suitable edge-detection parameters (manual, Otsu, Hossain et al.).
#
# IMPORTANT — MODEL LIMITATION:
# The segmentation step requires a *custom-trained YOLOv8 model* that was trained
# specifically on a unique physical steel-beam object. Without this exact model
# (not included here) and without the corresponding physical object, no valid
# segmentation results can be produced. The pipeline *cannot* be applied to other
# objects out-of-the-box.
#
# PURPOSE OF THIS PUBLIC VERSION:
# This script documents the computer-vision workflow used in the measurement
# system and can serve as a template for users who want to implement their own
# segmentation + edge-extraction pipeline. To re-use it, a new YOLO model trained
# on your target object is mandatory.
#
# RELATION TO TACHYMETER WORKFLOWS:
# This script does not contain any total-station control logic. It only provides
# the CV preprocessing steps (segmentation, cropping, Canny). The output is
# compatible with any downstream workflow — including total-station automation
# on Leica GeoCOM devices or instruments from other manufacturers — as long as
# the user embeds their own device-specific control layer.
#
# OUTPUT:
# - Cropped object image
# - Interactive Canny edge map
# - Bounding-box top-left corner coordinates
#
# Dependencies:
#   ultralytics, OpenCV, NumPy, Tkinter, ttkbootstrap, PIL
# -----------------------------------------------------------------------------



from ultralytics import YOLO  # Import the YOLO library for object detection and segmentation
import cv2  # Import OpenCV for image processing
import os  # Import os for file and directory operations
import numpy as np  # Import NumPy for numerical operations
import tkinter as tk  # Import Tkinter for GUI
import ttkbootstrap as ttk  # Für modernes Styling
from ttkbootstrap.constants import * # Import constants for ttkbootstrap styling
from PIL import Image, ImageTk  # Import PIL for image handling in Tkinter
import sys  # Import sys for system-specific parameters and functions


def interactive_canny_edge_detection(image, root, aperture_size=3, l2_gradient=False):
    """
    Opens an interactive GUI for Canny edge detection on the given image.
    Allows the user to adjust thresholds, Sobel filter size, and L2 gradient option,
    or use automatic thresholding methods (Otsu, Hossain et al.).
    Returns the selected lower/upper thresholds, aperture size, and L2 gradient flag.
    """    
    # Make a copy of the original image for later use
    original_image = image.copy()
    # Canny Edge Detection needs a grayscale image, so convert the image to grayscale
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Calculate the scale for resizing the image to fit in the GUI window
    orig_height, orig_width = gray.shape[:2]
    max_width, max_height = 800, 400  # Maximum dimensions for the GUI window, Could be adjusted as needed
    scale_w = max_width / orig_width
    scale_h = max_height / orig_height
    scale = min(scale_w, scale_h)
    window_width = int(orig_width * scale)
    window_height = int(orig_height * scale)

    # Create a new Tkinter window for the interactive Canny edge detection
    first_gui = tk.Toplevel(root)
    first_gui.title("Interactive Canny Edge Detection")
    first_gui.geometry(f"{max(window_width + 400, 1200)}x{max(window_height + 400, 900)}")
    first_gui.rowconfigure(0, weight=1)
    first_gui.columnconfigure(0, weight=1)

    # Tkinter variables for controls
    main_frame = ttk.Frame(first_gui, padding=20)
    main_frame.grid(row=0, column=0, sticky="nsew")
    main_frame.rowconfigure(0, weight=1)
    main_frame.columnconfigure(0, weight=1)

    # Set up variables for thresholds and Sobel filter size
    lower_var = ttk.IntVar(value=0)
    upper_var = ttk.IntVar(value=0)
    sobel_filter_var = ttk.IntVar(value=aperture_size)
    l2_gradient_var = ttk.BooleanVar(value=l2_gradient)

    # Set the default mode to "Manual Canny Edge Detection"
    mode_var = ttk.StringVar(value="Manual Canny Edge Detection")

    # Create a label to display the image
    image_label = ttk.Label(main_frame)
    image_label.grid(row=0, column=0, columnspan=3, padx=10, pady=20)

    def on_close():
        """
        Handle the window close event to clean up resources and exit the program.
        """
        # Destroy the window and exit the program when the window is closed
        first_gui.quit()
        first_gui.destroy()
        sys.exit(0)

    def update_canny(*args):
        """
        Update the Canny edge detection with the current threshold values and other parameters.
        """
        lower = lower_var.get()
        upper = upper_var.get()
        sobel_filter_size = sobel_filter_var.get()
        l2_gradient = l2_gradient_var.get()

        # Apply Canny edge detection with the current parameters
        canny = cv2.Canny(gray, lower, upper, apertureSize=sobel_filter_size, L2gradient=l2_gradient)

        # Update the image in the GUI
        display_image_for_gui(canny, image_label)

    def display_image_for_gui(img, label, max_width=800, max_height=400):
        """
        Display the given image in the Tkinter label, scaling it proportionally to fit the GUI.
        Converts grayscale to RGB if needed.
        """
        # If the image is grayscale, convert it to RGB for display
        if len(img.shape) == 2:
            img_disp = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img_disp = img.copy()

        # Resize the image to fit within the specified maximum dimensions
        orig_height, orig_width = img_disp.shape[:2]
        scale_w = max_width / orig_width
        scale_h = max_height / orig_height
        scale = min(scale_w, scale_h)
        new_width = int(orig_width * scale)
        new_height = int(orig_height * scale)
        img_disp = cv2.resize(img_disp, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)

        # Convert the image to a format suitable for Tkinter
        img_disp = Image.fromarray(img_disp)
        photo = ImageTk.PhotoImage(img_disp)
        label.config(image=photo)
        label.image = photo

    def canny_with_otsu(image, ratio=0.5):
        """
        Perform Canny edge detection using Otsu's method to determine thresholds.
        Returns the edge image and the computed lower/upper thresholds.
        """
        # Canny edge detection using Otsu's method to determine the thresholds
        # This function computes the Otsu threshold and applies Canny edge detection with a specified ratio 

        # Convert to grayscale if input is color
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        # Compute Otsu threshold
        _, otsu_thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Ensure otsu_thresh is a scalar
        if isinstance(otsu_thresh, np.ndarray):
            otsu_thresh = np.mean(otsu_thresh)

        # Compute lower and upper thresholds
        # The ratio is usually set to 0.5, but can be adjusted based on the image characteristics
        lower = int(otsu_thresh * ratio)
        upper = int(otsu_thresh)

        # Apply Canny edge detection with the computed thresholds
        edges = cv2.Canny(gray, threshold1=lower, threshold2=upper)

        # The edges and thresholds are returned for the right diplay in the GUI
        return edges, lower, upper

    def adaptive_canny(image):
        """
        Perform adaptive Canny edge detection based on the method in the paper "Dynamic Thresholding based Adaptive Canny Edge Detection" by Hossain et al.
        Returns the edge image and the computed lower/upper thresholds.
        """
        # Convert the image to grayscale if it is not already
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image

        # Step 1: Apply Gaussian filter to smooth the image
        kernel = np.array([
            [2,  4,  5,  4, 2],
            [4,  9, 12,  9, 4],
            [5, 12, 15, 12, 5],
            [4,  9, 12,  9, 4],
            [2,  4,  5,  4, 2]
        ], dtype=np.float32)
        kernel /= 159  # Normalize the kernel (Adjusted because of a printing error in the paper)
        blurred = cv2.filter2D(gray, -1, kernel)

        # Step 2: Compute gradients (Gx, Gy) and gradient magnitude
        # Gy and Gx are adjusted because of a printing error in the paper
        Gx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        Gy = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.hypot(Gx, Gy)

        # Step 3: Perform non-maximum suppression
        H, W = gradient_magnitude.shape
        nms = np.zeros((H, W), dtype=np.float32)
        for y in range(1, H - 1):
            for x in range(1, W - 1):
                theta = np.arctan2(Gy[y, x], Gx[y, x]) * 180 / np.pi
                if theta < 0:
                    theta += 180

                q = r = 0
                if (0 <= theta < 22.5) or (157.5 <= theta <= 180):
                    q = gradient_magnitude[y, x + 1]
                    r = gradient_magnitude[y, x - 1]
                elif 22.5 <= theta < 67.5:
                    q = gradient_magnitude[y - 1, x + 1]
                    r = gradient_magnitude[y + 1, x - 1]
                elif 67.5 <= theta < 112.5:
                    q = gradient_magnitude[y - 1, x]
                    r = gradient_magnitude[y + 1, x]
                elif 112.5 <= theta < 157.5:
                    q = gradient_magnitude[y - 1, x - 1]
                    r = gradient_magnitude[y + 1, x + 1]

                if gradient_magnitude[y, x] >= q and gradient_magnitude[y, x] >= r:
                    nms[y, x] = gradient_magnitude[y, x]

        # Step 4: Compute dynamic thresholds
        mu = np.mean(nms)  # Mean of the gradient magnitude
        sigma = np.var(nms)  # Variance of the gradient magnitude
        A = np.max(blurred)  # Maximum intensity in the grayscale image
        t_low = max(0, (mu - sigma) / 5) # The constant a is set to the lowest suggested value in the paper to avoid too low thresholds
        t_high = min(A, (mu + sigma) / 2) # The constant b is also set to the lowest suggested value in the paper to avoid too low thresholds



        # Step 5: Apply Canny edge detection using the computed thresholds
        edges = cv2.Canny(gray, threshold1=int(t_low), threshold2=int(t_high))

        # Opional: To apply the whole algorithm from the paper "Dynamic Thresholding based Adaptive Canny Edge Detection" use:
        # The method could handle heavy noise reduction better, because of "Step 1: Apply Gaussian filter to smooth the image",
        # because of the kernel size of 5x5 (The Canny edge detection algorithm uses a kernel size of 3x3 of default).
        # edges = cv2.Canny(blurred, threshold1=int(t_low), threshold2=int(t_high))
        # save_path = r"C:\Users\felix\Documents\Master\Masterarbeit\GeoCOM\Tachymeter_output_files\Dynamic_Thresholding_based_adaptive_Canny_Edge_Detection.jpg"
        # if cv2.imwrite(save_path, edges):
        #     print(f"Result image successfully saved to {save_path}.")
        # else:
        #     print("Error saving the result image.")

        # The edges and thresholds are returned for the right diplay in the GUI
        return edges, int(t_low), int(t_high)  

    def switch_mode(*args):
        """
        Switch between manual and automatic Canny edge detection modes.
        Updates the GUI and image preview accordingly.
        """
        # Switch between manual and automatic modes based on the selected option
        # Get the selected mode from the dropdown menu
        mode = mode_var.get()

        # After the mode is changed, update the image with the default parameters
        lower_var.set(0)
        upper_var.set(0)
        l2_gradient_var.set(False)
        sobel_filter_var.set(3)

        # Settings for the manual mode
        if mode == "Manual Canny Edge Detection":
            # Reset the Sobel filter size to 3
            sobel_filter_var.set(3)

            # Show the GUI elements for the manual mode on the right position and with the right visualization
            lower_slider.grid(row=3, column=1, sticky="ew", padx=5, pady=10)
            upper_slider.grid(row=4, column=1, sticky="ew", padx=5, pady=10)
            lower_value_label.grid(row=3, column=2, sticky="w", padx=5, pady=10)
            upper_value_label.grid(row=4, column=2, sticky="w", padx=5, pady=10)
            sobel_label.grid(row=5, column=0, sticky="w", padx=10, pady=10)
            sobel_menu.grid(row=5, column=1, sticky="ew", padx=5, pady=10)
            sobel_menu.configure(width=20) 
            l2_gradient_checkbox.grid(row=6, column=0, columnspan=3, pady=20)

            # Remove the labels for the automatic mode
            lower_threshold_label.grid_remove()
            upper_threshold_label.grid_remove()

            # Update the Canny edge detection with the current parameters
            # This is necessary to ensure that the image is displayed correctly when switching back to manual mode
            update_canny()

        else:
            # Hide the GUI elements for the manual mode
            lower_slider.grid_remove()
            upper_slider.grid_remove()
            lower_value_label.grid_remove()
            upper_value_label.grid_remove()
            sobel_label.grid_remove()
            sobel_menu.grid_remove()
            l2_gradient_checkbox.grid_remove()

            # If the automatic mode is selected, the thresholds are calculated with call of the function
            # and the image is updated with the new parameters
            if mode == "Automatic Canny Edge Detection with Otsu":
                edges, lower, upper = canny_with_otsu(original_image)
                lower_var.set(lower)
                upper_var.set(upper)
            elif mode == "Automatic Canny Edge Detection with Hossain et al. method":
                edges, lower, upper = adaptive_canny(original_image)
                lower_var.set(lower)
                upper_var.set(upper)

            # Scales the edges and actualizes the image in the GUI
            display_image_for_gui(edges, image_label)

            # Show the calculated thresholds in the GUI on the right position and with the right visualization
            lower_threshold_label.config(text=lower_var.get())
            upper_threshold_label.config(text=upper_var.get())
            lower_threshold_label.grid(row=3, column=1, sticky="w", padx=5, pady=10)
            upper_threshold_label.grid(row=4, column=1, sticky="w", padx=5, pady=10)

    # Variable to store the returned thresholds
    thresholds = []

    def return_thresholds():
        """
        Collects the selected thresholds and parameters, closes the GUI,
        and returns them to the calling function.
        """
        print("Returning thresholds and parameters...")
        # Print the thresholds and additional parameters
        print(f"Lower Threshold: {lower_var.get()}")
        print(f"Upper Threshold: {upper_var.get()}")
        print(f"Aperture Size: {sobel_filter_var.get()}")
        print(f"L2 Gradient: {l2_gradient_var.get()}")

        # Store the thresholds and additional parameters in the list
        thresholds.append(lower_var.get())
        thresholds.append(upper_var.get())
        thresholds.append(sobel_filter_var.get())
        thresholds.append(l2_gradient_var.get())

        # Close the Toplevel window
        first_gui.quit() # Quit the main loop
        first_gui.destroy() # Destroy the Toplevel window

    # Initialize the image with the default parameters
    update_canny()

    # Create the GUI Dropdown menu for the Canny edge detection mode
    ttk.Label(main_frame, text="Canny Edge Detection Mode", font=("Helvetica", 10, "bold")).grid(row=1, column=0, sticky="w", padx=10, pady=10)
    mode_menu = ttk.OptionMenu(main_frame, mode_var, mode_var.get(), 
                           "Manual Canny Edge Detection", 
                           "Automatic Canny Edge Detection with Otsu", 
                           "Automatic Canny Edge Detection with Hossain et al. method", 
                           command=lambda *args: switch_mode(*args))
    mode_menu.grid(row=1, column=1, sticky="ew", padx=5, pady=10)

    # Create the GUI elements for the manual mode
    # Slider for lower threshold
    ttk.Label(main_frame, text="Lower Threshold", font=("Helvetica", 10, "bold")).grid(row=3, column=0, sticky="w", padx=10, pady=10)
    lower_slider = ttk.Scale(main_frame, from_=0, to=255, orient="horizontal", variable=lower_var, command=update_canny, bootstyle="info")
    lower_slider.grid(row=3, column=1, sticky="ew", padx=5, pady=10)
    lower_slider.configure(length=400)

    # Dynamic display of the lower threshold value
    # The label is updated with the current value of the lower threshold
    lower_value_label = ttk.Label(main_frame, text=f"{lower_var.get()}", font=("Helvetica", 10))
    lower_value_label.grid(row=3, column=2, sticky="w", padx=5, pady=10)
    lower_var.trace_add("write", lambda *args: lower_value_label.config(text=f"{lower_var.get()}"))

    # Create the GUI elements for the manual mode
    # Slider for upper threshold
    ttk.Label(main_frame, text="Upper Threshold", font=("Helvetica", 10, "bold")).grid(row=4, column=0, sticky="w", padx=10, pady=10)
    upper_slider = ttk.Scale(main_frame, from_=0, to=255, orient="horizontal", variable=upper_var, command=update_canny, bootstyle="info")
    upper_slider.grid(row=4, column=1, sticky="ew", padx=5, pady=10)
    upper_slider.configure(length=400)

    # Dynamic display of the upper threshold value
    # The label is updated with the current value of the upper threshold
    upper_value_label = ttk.Label(main_frame, text=f"{upper_var.get()}", font=("Helvetica", 10))
    upper_value_label.grid(row=4, column=2, sticky="w", padx=5, pady=10)
    upper_var.trace_add("write", lambda *args: upper_value_label.config(text=f"{upper_var.get()}"))

    # Sobel Filter Size Dropdown for manual mode
    # The Sobel filter size is set to 3 by default
    sobel_label = ttk.Label(main_frame, text="Sobel Filtersize", font=("Helvetica", 10, "bold"))
    sobel_label.grid(row=5, column=0, sticky="w", padx=10, pady=10)
    sobel_menu = ttk.OptionMenu(main_frame, sobel_filter_var, sobel_filter_var.get(), 3, 5, 7, command=update_canny)
    sobel_menu.grid(row=5, column=1, sticky="ew", padx=5, pady=10)

    # Create the checkbox for L2 gradient option
    # The checkbox is set to False by default
    l2_gradient_checkbox = ttk.Checkbutton(main_frame, text="Use L2 Gradient", variable=l2_gradient_var, command=update_canny)
    l2_gradient_checkbox.grid(row=6, column=0, columnspan=3, pady=20)

    # Create labels for the automatic mode to display the calculated thresholds
    lower_threshold_label = ttk.Label(main_frame, text="", font=("Helvetica", 10))
    upper_threshold_label = ttk.Label(main_frame, text="", font=("Helvetica", 10))

    # Create the button to return the thresholds and close the GUI
    ttk.Button(main_frame, text="Return Thresholds", command=return_thresholds, bootstyle="success").grid(row=7, column=0, columnspan=3, pady=20)
    first_gui.protocol("WM_DELETE_WINDOW", on_close)

    # Bind the mode variable to the switch_mode function to update the GUI when the mode changes
    first_gui.mainloop()

    # Return the thresholds after the GUI is closed
    return tuple(thresholds)

def process_images_from_another_script(model_path, input_folder, output_folder, root):
    """
    Processes all images in the input folder using YOLOv8 for object segmentation.
    For each detected object, crops the object, runs interactive Canny edge detection,
    and saves the edge image, cropped image, and bounding box coordinates to the output folder.
    In the steelbeam workflow, there is always only one image in the folder.
    """  

    # Create the output folder if it doesn't already exist
    os.makedirs(output_folder, exist_ok=True)  

    # Load the YOLO model
    model = YOLO(model_path)

    # Iterate through all images in the input folder (in the steelbeam workflow, there is always only one image in the folder)
    for image_file in os.listdir(input_folder):
        if image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(input_folder, image_file)
            img = cv2.imread(image_path)
            H, W, _ = img.shape  # Get the height and width of the image

            # Perform prediction using the YOLOv8 model
            results = model(img)

            for result in results:
                # Check if any objects were detected
                if result.masks is None or result.boxes is None or len(result.boxes) == 0:
                    print(f"[ERROR] No objects detected in image: {image_file}")
                    sys.exit(1)

                # Print the confidence for each detected object
                for i, box in enumerate(result.boxes):
                    conf = float(box.conf[0]) if hasattr(box, 'conf') else None
                    print(f"[INFO] Object {i+1} confidence: {conf:.2f}" if conf is not None else "[INFO] Confidence not available.")

                # Process each mask from the YOLO results
                for j, mask in enumerate(result.masks.data):
                    # Convert the mask to a NumPy array and scale it to the range [0, 255]
                    mask = mask.numpy() * 255

                    # Resize the mask to match the original image dimensions
                    mask = cv2.resize(mask, (W, H)).astype(np.uint8)

                    # Binarize the mask using a threshold
                    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

                    # Add padding to the mask
                    # The formular to calculate the padding size is: (kernel_size - 1) // 2
                    # To calculate the kernel size, we can use the formula: kernel_size = 2 * padding + 1
                    # Define the padding in pixels
                    padding_pixels = 0  # Set the padding size in pixels; it is set to 0 by default, because the padding does effect double edges
                    # Create a kernel for dilation with the specified padding size
                    kernel_size = 2 * padding_pixels + 1  
                    kernel = np.ones((kernel_size, kernel_size), np.uint8)  
                    padded_mask = cv2.dilate(mask, kernel, iterations=1)

                    # Create a masked image where everything outside the padded mask is black
                    masked_img = cv2.bitwise_and(img, img, mask=padded_mask)

                    # Find the bounding box of the padded mask
                    contours, _ = cv2.findContours(padded_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        largest_contour = max(contours, key=cv2.contourArea)
                        x, y, w, h = cv2.boundingRect(largest_contour)
                        top_left_corner = (x, y)
                        print(f"Upper left corner of the mask with padding: {top_left_corner}")

                        # Crop the masked image to the bounding box
                        cropped_img = masked_img[y:y+h, x:x+w]

                        # Start the interactive Canny Edge Detection
                        lower_thresh, upper_thresh, aperture_size, l2_gradient = interactive_canny_edge_detection(cropped_img, root)

                        # Apply Canny Edge Detection with selected thresholds and parameters
                        edges = cv2.Canny(cropped_img, lower_thresh, upper_thresh, apertureSize=aperture_size, L2gradient=l2_gradient)

                        edge_output_path = os.path.join(output_folder, f"edges_{image_file}")
                        # Delete the fileextension from the image file to create a base name for the output files
                        image_base_name = os.path.splitext(image_file)[0]

                        # Save the edge-detected image and the upper left corner coordinates of the bounding box
                        corner_output_path = os.path.join(output_folder, f"left_upper_corner_BB_{image_base_name}.txt")
                        cropped_image_path = os.path.join(output_folder, f"cropped_{image_file}")
                        with open(corner_output_path, "w") as f:
                            f.write(f"{top_left_corner}\n")
                        cv2.imwrite(edge_output_path, edges)
                        cv2.imwrite(cropped_image_path, cropped_img)


if __name__ == "__main__":
    """
    Main entry point: sets up paths, initializes the Tkinter root,
    and starts the processing workflow.
    """

    # Define paths
    # Make sure to use your own pretrained Yolov8-Model
    model_path = r'.\last.pt'
    input_folder = r'.\Testimages'
    output_folder = r'.\results5'

    # Create the main Tk instance and apply darkly style
    root = tk.Tk()
    root.withdraw()  # Hide the main Tk window
    style = ttk.Style("darkly")

    # Call the main processing function
    process_images_from_another_script(model_path, input_folder, output_folder, root)