# -----------------------------------------------------------------------------
# Public / sanitized version of the original main_total_station_control.py
#
# This file provides the full workflow structure used in the associated
# master's thesis of Felix Staab: "Automated Object Measurement with a 
# Total Station Using Instance Segmentation"
#
# The original implementation controlled a Leica MS50 using GeoCOM. All concrete
# GeoCOM command strings and Leica-specific ASCII request formats have been
# removed due to licensing restrictions. Every operation that originally triggered
# a real device command is now replaced by a descriptive placeholder.
#
# WHAT IS GEOCOM?
# GeoCOM is Leica Geosystems' proprietary command interface for programmatic
# control of robotic total stations (e.g., camera control, telescope positioning,
# EDM measurement, file transfer). A valid implementation requires the official
# Leica GeoCOM manual, which is not included in this repository.
#
# IMPORTANT:
# - This script does NOT control a real total station out of the box.
#   You must implement your own communication layer and map the placeholder
#   strings to valid device commands in a private module.
#
# - This pipeline is not limited to Leica hardware. Other total-station
#   manufacturers usually provide similar command interfaces (e.g., Trimble,
#   Topcon, Sokkia). The structure of this script can be reused, but the
#   concrete command syntax must be replaced according to the vendor's API.
#
# - The Computer Vision pipeline (YOLO segmentation → Canny → Hough → Harris/DBSCAN)
#   cannot segment any meaningful object unless BOTH conditions are met:
#       1) You provide the custom YOLOv8 model trained in the thesis.
#       2) You have access to the physical measurement object the model was
#          trained for. The object is a one-off prototype; no generic object
#          can be segmented.
#
# PURPOSE OF THIS PUBLIC VERSION:
# - Provide researchers and developers with a clean reference of the full
#   automation workflow (image acquisition → CV processing → pixel-based
#   targeting → measurement) without exposing confidential command syntax.
# - Allow reproduction of the high-level logic on any total station,
#   provided the user supplies a compatible device API.
# - Serve as a structural template for implementing your own automated
#   measurement system on Leica MS50 or alternative instruments.
#
# All comments and docstrings were rewritten and shortened for publication.
# -----------------------------------------------------------------------------



import serial  #This module encapsulates the access for the serial port
import math    #This module provides access to the mathematical functions defined by the C standard
import time    #This module provides various time-related functions
from datetime import datetime  # Import the datetime class from the datetime module
import logging #This module is for a better debugging to unterstand the tachymeter commands and responses
import os      #This module provides a way of using operating system dependent functionality
from PIL import Image  #This module is used to open, manipulate and save many different image file formats
import matplotlib.pyplot as plt  #This module is used to create a figure, create a plotting area in a figure, plot some lines in a plotting area, decorate the plot with labels, etc.
import socket  # Import the socket module to use network connections
import re     #This module provides regular expression matching operations
import matplotlib.patches as patches #This module is used to add shapes to the plot
import sys #This module provides access to some variables used or maintained by the interpreter and to functions that interact with the interpreter
import cv2 #This module is used for computer vision tasks, such as image processing and analysis
import tkinter as tk
from ttkbootstrap import Style, ttk
from ttkbootstrap.constants import *
import shutil #This module is used to perform high-level operations on files and collections of files
sys.path.append() # Add the path to the folder containing the external "Object_Segmentation_and_Canny_Edge_Detection_App.py" script
from Object_Segmentation_and_Canny_Edge_Detection_App import process_images_from_another_script # Import the function to process image segmentation and Canny edge detection
sys.path.append() # Add the path to the folder containing the external "Hough_Line_Transform_App.py" and the "Harris_Corner_detection_and_DBSCAN_Clustering_App.py" script
from Hough_Line_Transform_App import run_Hough_from_another_script  # Impport the function to process images with Hough Line Transform
from Harris_Corner_detection_and_DBSCAN_Clustering_App import run_harris_gui_from_another_script # Import the function to process images with Harris Corner Detection and DBSCAN-Clustering


# ================================
# 🔹 Automation functions
# ================================
"""
Automation functions for controlling key robotic features of the tachymeter.
These functions allow programmatic positioning of the instrument, including:
- Moving the telescope to a specific pixel coordinate in the camera image.
"""

def TelescopeToPixelCoord():
    """
    Generate a GeoCOM command to position the theodolite so that the crosshairs
    are aligned with the specified pixel coordinates in the selected camera image.

    """
    Request = ""
    return Request  


# ================================
# 🔹 Camera functions
# ================================
"""
Camera functions for controlling and querying the cameras of the tachymeter.
All functions in this section allow you to:
- Set image properties and capture images
- Get crosshair positions
"""

def Is_CAM_Ready():
    """
    Generate a GeoCOM command to check if the specified camera is ready for use.

    """
    Request = ""
    return Request

def SetActualImageName(szName, lNumber):   
    """
    Generate a GeoCOM command to set the image name and number for the next image capture.

    """
    Request = f"{szName}, {lNumber}"
    return Request

def OVC_TakeImage():    
    """
    Generate a GeoCOM command to capture an image with the overview camera.
    The image will be saved with the name set by SetActualImageName.

    """
    Request = ""
    return Request

def OVC_GetCameraCentre():  
    """
    Generate a GeoCOM command to calculate the crosshair position of the overview camera (OVC)
    at the current distance and resolution.

    """
    Request = ""
    return Request

def SetCameraProperties():  
    """
    Generate a GeoCOM command to set image resolution and compression for the next image.
    """

    Request = ""
    return Request

def OVC_SetActDistance():   
    """
    Generate a GeoCOM command to set the distance to the current target for the overview camera (OVC).
    This affects the camera centre calculation.

    """
    Request = ""
    return Request

def SetWhiteBalanceMode():  
    """
    Generate a GeoCOM command to set the white balance mode of the camera.

    """
    Request = ""
    return Request


# ================================
# 🔹 Communication functions
# ================================
"""
Communication functions for managing the connection between the PC and the tachymeter.
All functions in this section allow you to:
- Send commands and receive responses via serial port or WLAN/TCP

These functions abstract the details of the underlying communication protocol and provide a unified interface for command exchange.
"""

def COM_send_serial_command(command):   
    """
    Send a GeoCOM command to the tachymeter via the serial port and return its response.
    The command and response are also logged for traceability.

    Parameters:
        command (str): The GeoCOM ASCII command string to send

    Returns:
        str: The response received from the tachymeter (decoded as UTF-8)
    """
    # Open the serial port with the specified settings
    serialPort = serial.Serial(port="COM4", baudrate=115200, bytesize=8, stopbits=serial.STOPBITS_ONE, timeout=10)
    # Send the command (encoded as UTF-8)
    serialPort.write(command.encode('UTF-8'))
    # Read the response until the end of the line (terminated by \r\n)
    response = serialPort.read_until(b'\r\n').decode('UTF-8')
    # If no response is received, set a default message
    if not response:
        response = "No response received."
    # Log the command and response
    COM_log_tachymeter(command, response)
    return response

def COM_send_WLAN_command(command): 
    """
    Send a GeoCOM command to the tachymeter via WLAN/TCP and return its response.
    The command and response are also logged for traceability.

    Parameters:
        command (str): The GeoCOM ASCII command string to send

    Returns:
        str or None: The response received from the tachymeter, or None if an error occurred
    """
    # IP address and port of the tachymeter
    TACHYMETER_IP = "192.168.178.42"
    TACHYMETER_PORT = 1212  # Use the LUPA port (554). The LUPA port is the standard port for communication with the tachymeter. Additional there is a RTSP port for video streaming with the number 554.
    BUFFER_SIZE = 4096  # Increased buffer size for large responses
    TIMEOUT = 10  # Increased timeout to prevent lost responses
    try:
        # Create a socket object for the connection
        # socket.AF_INET: Uses IPv4 addresses
        # socket.SOCK_STREAM: Uses TCP for reliable, connection-oriented communication
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:    # 'with' ensures that the socket is properly closed after use
            s.settimeout(TIMEOUT)  # Set a timeout of 5 seconds for the connection
            s.connect((TACHYMETER_IP, TACHYMETER_PORT))  # Establish a connection to the tachymeter
            print(f"[LOG] ✅ Successfully connected to the tachymeter with IP {TACHYMETER_IP} and Port {TACHYMETER_PORT}!")

            # Send the ASCII command
            s.sendall(command.encode())  # Send the command to the tachymeter (as a byte string)

            # Receiving the full response from the tachymeter
            response = ""
            while True:
                # Receive a chunk of data from the socket
                chunk = s.recv(BUFFER_SIZE).decode()
                # Append the received chunk to the response
                response += chunk
                # Check if the chunk is empty or if the response ends with the GeoCOM message terminator "\r\n"
                if not chunk or response.endswith("\r\n"):  # End of GeoCOM message
                    break

            # Log the command and the response
            COM_log_tachymeter(command, response)

            return response  # Return the response from the tachymeter

    except Exception as e:
        # Error handling in case the connection fails or another error occurs
        # 'e' contains the exception instance with the error message
        print(f"❌ Connection error: {e}")
        return None  # Return None if there was an error


# ================================
# 🔹 EDM functions
# ================================
"""
EDM (Electronic Distance Measurement) functions for controlling the distance measurement hardware of the tachymeter.
All functions in this section allow you to:
- Switch the laser pointer on or off
"""

def Laserpointer(state):
    """
    Generate a GeoCOM command to switch the laser pointer of the tachymeter on or off.

    Parameters:
        state (int): 1 to turn the laser pointer ON, 0 to turn it OFF

    """
    Request = state
    return Request


# ================================
# 🔹 File Transfer functions
# ================================
"""
File Transfer functions for listing, downloading, and deleting image files on the tachymeter.
All functions in this section allow to:
- Download files block-wise
- Delete files
"""

def AbortDownload():    
    """
    Generate a GeoCOM command to abort or end the file download command.

    """
    Request = ""
    return Request

def SetupDownloadLarge():   
    """
    Generate a GeoCOM command to set up the download for large files from the instrument.
    Must be called before DownloadXL.

    """
    Request = ""
    return Request

def DownloadXL():   
    """
    Generate a GeoCOM command to get a single block of data for large file download.
    SetupDownloadLarge must be called first.

    """
    Request = ""
    return Request

def Delete(FileName):   
    """
    This command deletes one or more files. 
    """

    Request = FileName    
    return Request

# ================================
# 🔹 Theodolite measurement and calculation functions
# ================================
"""
Theodolite measurement and calculation functions for controlling and querying the measurement hardware of the tachymeter.
All functions in this section allow you to:
- Trigger and control angle and distance measurements
- Retrieve corrected or uncorrected values for angles, distances, and position coordinates
- Query measurement results in various modes (single, tracking, with/without inclination correction)
"""

def GetCoordinate():    
    """
    Generate a GeoCOM command to query an angle measurement and an inclination measurement. 
    Calculates the coordinates of the measured point using a previously measured distance.
    A distance measurement must be started in advance.

    """
    Request = ""
    return Request

def DoMeasure():    
    """
    Generate a GeoCOM command to carry out a distance measurement.
    This command does not return the measured values directly.

    """
    Request = ""
    return Request 

def GetDistance():  
    """
    Generate a GeoCOM command to return the angles and distance measurement data.
    This command does not trigger a new distance measurement; a measurement must be started in advance.
    If a valid distance measurement is available the results are returned.

    """
    Request = ""
    return Request


# ================================
# 🔹 General functions 
# ================================
"""
General helper and workflow functions for the tachymeter control script.
These functions do not send GeoCOM commands directly, but provide essential utilities for:
- Logging communication with the tachymeter
- File management (search, download, verification, deletion, moving)
- Image processing workflow orchestration
- Coordinate adjustment and crosshair drawing
- Folder cleanup and utility routines

They are used to support and automate the measurement and image processing workflow.
"""

def COM_log_tachymeter(command, response):  
    """
    Log a message related to tachymeter communication, including the sent command and received response.
    The log is written to 'Tachymeter.log' in the current working directory and also printed to the console.

    Parameters:
        command (str): The GeoCOM command sent to the tachymeter.
        response (str): The response received from the tachymeter.
    """
    # Settings for logging
    # os.getcwd() returns the current working directory, where the script is executed. The log file will be created in this directory.
    log_file_path = os.path.join(os.getcwd(), "Tachymeter.log")  

    # If the log file does not exist, it will be created
    if not os.path.exists(log_file_path):
        with open(log_file_path, "a") as log_file:  # 'a' stands for append. That means that the file isn't overwritten and the new log is added to the existing log
            log_file.write("Log started\n")         # 'w' would overwrite the file

    #The following log is used to log the commands and responses of the tachymeter
    logging.basicConfig(filename = log_file_path, level = logging.INFO, format = '%(asctime)s - %(message)s')

    #If the code is in development or errors must be found in more detail, the following log could be more useful
    #logging.basicConfig(filename = log_file_path, level = logging.DEBUG, format = '%(asctime)s:%(levelname)s:%(message)s')

    # The log message is formatted as a block containing both the sent command and received response
    log_message = ("\n"
        "==================================================================================================\n"
        f"[LOG] PC sent command: {command.strip()}\n"
        "--------------------------------------------------------------------------------------------------\n"
        f"[LOG] PC received response: {response.strip()}\n"
        "==================================================================================================\n\n"
    )

    logging.info(log_message)
    #The log message is also printed to the console
    print(log_message)

def downloadXL_and_delete_file(file_name):        
    """
    Download a large file from the tachymeter block-wise and delete it after successful transfer.

    Steps:
    1. Ensure the correct file format (.jpg).
    2. Send a setup command to initiate the download.
    3. Parse the response to extract the number of blocks required.
    4. Download the file block by block.
    5. Delete the file from the tachymeter once the transfer is complete.

    """

    print(f"[LOG] Initiating download of: {file_name}")

    # Step 1: Ensure correct filename format
    if not file_name.lower().endswith(".jpg"):  # endswith() method returns True if the string ends with .jpg, otherwise False
        file_name += ".jpg"  # Append .jpg extension if missing

    # Step 2: Send setup command to initialize file download
    setup_command = SetupDownloadLarge()
    response = COM_send_WLAN_command(setup_command)

    # Step 3: Parse the response to extract the number of required blocks
    response_parts = response.split(":")[-1].split(",")  # Extract relevant response part
    response_code = response_parts[0].strip()  # First value in response is the status code

    # Define known error codes of the FTR_SetupDownload funktion and their meanings in a dictionary
    error_codes = {
        "0": "Execution successful.",
        "2": "Device not available or cannot get path.",
        "26": "Setup already done or AbortDownload() not called.",
        "13059": "Block size too big.",
        "13056": "File access error."
    }

    # Step 3a: Handle potential errors from SetupDownload response
    if response_code != "0":  # If response code is not "0", an error occurred
        print(f"[ERROR] {error_codes.get(response_code, 'Unknown response code')}")  # Search for error code in dictionary, otherwise print 'Unknown response code'
        return False

    # Step 3b: Extract the number of required blocks for the file transfer
    num_blocks = int(response_parts[1].strip())  # Convert block count to integer
    print(f"[LOG] Number of blocks required: {num_blocks}")

    # Step 4: Download the file block by block
    with open(file_name, "wb") as file:  # Open file in write-binary mode and saves in the same directory as the script
        for block_num in range(1, num_blocks + 1):  # Loop through all blocks
            command = DownloadXL()  # Request specific block from tachymeter
            response = COM_send_WLAN_command(command).strip()  # Send command and get response
            print("Tachymeter response (unextracted):", response)
            # Extract response parts
            response_parts = response.split(":")[-1].split(",")  # Remove header and split
            print(f"[DEBUG] Tachymeter response (extracted) for block {block_num}: {response_parts}")
            # Define error codes in a dictionary
            error_codes = {
                "0": "Execution successful.",
                "13060": "Missing setup. Call SetupDownload() first.",
                "13059": "Invalid input. First block is missing or wrong order.",
                "13056": "File access error."
            }

            # Check for errors
            response_code = response_parts[0].strip()  # First value is the return code
            if response_code != "0":  # If not successful
                print(f"[ERROR] Block {block_num} failed: {error_codes.get(response_code, 'Unknown error')}")
                return False  # Stop downloading if there's an error

            # Check if the response is valid
            if len(response_parts) < 2:
                print(f"[ERROR] Invalid response format for block {block_num}")
                return False  # Stop if response is incorrect

            hex_data = response_parts[1].strip()
            print("Hex data:", hex_data)
            print(len(hex_data))
            try:
                # Remove non-hexadecimal characters from the hex_data
                #hex_data = ''.join(filter(lambda x: x in '0123456789abcdefABCDEF', hex_data))
                #hex_data = re.sub(r'[^0-9a-fA-F]', '', hex_data)

                block_data = bytes.fromhex(hex_data)  # Convert hex string to binary
            except ValueError as e:
                print(f"[ERROR] Failed to convert hex data to binary: {e}")
                return False  # Stop if conversion fails

            # Write block data to file
            file.write(block_data)  # Append binary data to file
            print(f"[LOG] Block {block_num}/{num_blocks} received successfully.")

    print(f"[SUCCESS] Download completed: {file_name}")
    

    # Step 5: Delete the file from the tachymeter after successful download
    print(f"[LOG] Deleting {file_name} from the tachymeter...")
    delete_response = COM_send_WLAN_command(Delete(file_name))  # Send delete command

    # Step 5a: Verify deletion response from the tachymeter
    delete_response  = delete_response.split(":")[-1].split(",")
    delete_response = delete_response[0].strip()
    if delete_response == "0":  # Successful deletion response
        print(f"[SUCCESS] File {file_name} successfully deleted from the device.")
    else:  # If response is not as expected, deletion failed
        print(f"[ERROR] Failed to delete {file_name}. Response: Device not available or can not get path.")

    return True  # Return True if both download and deletion were successful

def adjust_coordinates(bb_file, corners_file):  
    """
    Adjusts the local coordinates from Harris corner detection to the global coordinate system of the tachymeter image,
    based on the offset from the bounding box file.

    :param bb_file: Path to the file containing the upper left corner of the bounding box (as a tuple, e.g., (x, y)).
    :param corners_file: Path to the file containing the local pixel coordinates (each as a tuple, e.g., (x, y)).
    :return: List of adjusted global coordinates (as tuples).
    """
    # Read the upper left corner of the bounding box from the first file
    with open(bb_file, "r") as f:
        bb_offset = eval(f.readline().strip())  # Convert the string "(x, y)" to a tuple
        x_offset, y_offset = bb_offset

    # Read the local coordinates from the second file
    with open(corners_file, "r") as f:
        local_corners = [eval(line.strip()) for line in f.readlines()]  # Convert each line to a tuple

    # Calculate the global coordinates by adding the offset to each local coordinate
    global_corners = [(x + x_offset, y + y_offset) for x, y in local_corners]

    return global_corners

def delete_jpg_files(folder_path):      
    """
    Deletes all files with the .jpg extension in the specified folder.

    :param folder_path: Path to the folder to be searched for .jpg files.
    """
    # Check if the folder exists
    if not os.path.exists(folder_path):
        print(f"[ERROR] The folder {folder_path} does not exist.")
        return

    # Iterate through all files in the folder
    for file_name in os.listdir(folder_path):
        # Check if the file has a .jpg extension
        if file_name.lower().endswith(".jpg"):
            file_path = os.path.join(folder_path, file_name)
            try:
                # Delete the file
                os.remove(file_path)
                print(f"[INFO] Deleted file: {file_path}")
            except Exception as e:
                print(f"[ERROR] Error deleting file {file_path}: {e}")

def move_jpg_file(source_folder, target_folder, file_name): 
    """
    Moves a .jpg file from the source folder to the target folder.
    If the file does not exist in the source folder, nothing happens.
    """
    # Ensure the file name ends with .jpg
    if not file_name.lower().endswith(".jpg"):
        file_name += ".jpg"

    source_path = os.path.join(source_folder, file_name)
    target_path = os.path.join(target_folder, file_name)

    if not os.path.exists(source_path):
        print(f"[ERROR] Source file does not exist: {source_path}")
        return

    # Create target folder if it does not exist
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    try:
        shutil.move(source_path, target_path)
        print(f"[INFO] File moved to: {target_path}")
    except Exception as e:
        print(f"[ERROR] Could not move file: {e}")

def image_processing_tachymeter(image_name):    
    """
    Processes a single image (captured by the tachymeter) to extract object corners using a systematic workflow:
    1. Moves the image to the processing folder.
    2. Segments the main object using a YOLOv8 segmentation model and applies Canny edge detection.
    3. Detects lines using the Hough Line Transform.
    4. Detects corners using Harris corner detection and DBSCAN clustering.

    All intermediate and result files are saved in the specified output folder.

    Parameters:
        image_name (str): Name of the image file (without extension) to process.

    Returns:
        None
    """
    # Define input and output folders for image processing
    image_path = r'.\Python_Scripts'
    output_folder = r'.\Tachymeter_Outputfiles'

    # Step 1: Move the image to the processing folder to ensure a clean workspace
    move_jpg_file(image_path, output_folder, image_name)

    # Step 2: Create the main Tk instance for GUI-based steps (hidden window)
    root = tk.Tk()
    root.withdraw()  # Hide the main Tk window

    # Apply the darkly style for consistent GUI appearance
    style = Style("darkly")

    # Step 3: Image Segmentation and Canny Edge Detection
    # Make sure to use your own trained Yolov8-Model
    model_path = r'.\last.pt'
    input_folder = output_folder

    # Segment the main object and apply Canny edge detection
    # Make sure to connect the "Object_Segmentation_and_Canny_Edge_Detection_App.py" script properly correctly to use this function
    process_images_from_another_script(model_path, input_folder, output_folder, root)

    # Step 4: Hough Line Detection
    input_image_path = fr'.\Tachymeter_Outputfiles\edges_{image_name}.jpg'
    image_after_canny = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)

    # Detect lines using Hough transform
    # Make sure to connect the "Hough_Line_Transform_App.py" script properly correctly to use this function
    run_Hough_from_another_script(image_after_canny, root)

    # Step 5: Harris Corner Detection
    # Use the output from the Hough transform as input for corner detection
    image_path = r".\Hough_Line_Transform_Output.jpg"
    img_output_path = fr".\Harris_corner_detection_{image_name}.jpg"
    txt_output_path = fr".\Harris_corner_detection_coordinates_{image_name}.txt"

    # Detect corners and save results
    # Make sure to connect the "Harris_Corner_Detection_with_DBSCAN_App.py" script properly correctly to use this function
    run_harris_gui_from_another_script(image_path, img_output_path, txt_output_path, root)

def Take_Image_with_OVC_and_turn_telescope_to_pixel_coordinates(param=0):   
    """
    Automated workflow for capturing an image with the overview camera (OVC), processing the image,
    and moving the tachymeter telescope to the detected object corners.

    Process overview:
    1. Preparation: Deletes old images in the working and processing folders to ensure a clean workspace.
    2. Camera readiness: Checks if the OVC is ready for use.
    3. Image naming: Sets a unique image name for the next capture based on the current time.
    4. Camera configuration: Sets white balance and camera properties (resolution, compression, etc.).
    5. Distance measurement: Starts and reads a distance measurement to the object.
    6. Image acquisition: Captures an image with the OVC.
    7. Image transfer: Downloads the captured image to the PC and deletes it from the tachymeter.
    8. Image processing: Segments the object, detects edges, lines, and corners (including DBSCAN clustering).
    9. Coordinate adjustment: Converts local pixel coordinates to global image coordinates.
    10. Telescope control:
        - param=0: The telescope moves to each detected corner and the laser pointer is activated for visual feedback.
        - param=1: The telescope moves to each detected corner, performs a measurement, and saves the coordinates.
    11. Result: The measured coordinates are saved to a text file.

    Parameters:
        param (int): 
            0 = Demonstration mode (laser pointer only)
            1 = Measurement mode (measure and save coordinates)

    Notes:
        - All coordinates are rounded to 4 decimal places.
        - Extensive logging and error handling are included for each step.
    """
    # --- Step 1: Prepare folders for image processing ---
    # Define the folder where the OVC images are saved on the PC.
    ovc_image_folder = r""
    # Delete all .jpg files in the OVC image folder to avoid confusion with old images.
    delete_jpg_files(ovc_image_folder)

    # Define the folder for further image processing.
    processing_folder = r".\Tachymeter_Outputfiles"
    # Delete all .jpg files in the processing folder to ensure a clean workspace.
    delete_jpg_files(processing_folder)

    # --- Step 2: Ensure the OVC is ready ---
    COM_send_WLAN_command(Is_CAM_Ready())
    time.sleep(1)

    # --- Step 3: Set the image name for the next capture ---
    szName = "Picture_"  # Image prefix
    # Use the current time (HHMM) to ensure unique image names.
    lNumber = int(datetime.now().strftime("%H%M"))
    # Make sure the image name is globally accessible. 
    global image_name
    # Construct the image name using the prefix and current time.
    image_name = f'{szName}0{lNumber}'
    # Send command to set the image name for the next capture.
    COM_send_WLAN_command(SetActualImageName(szName, lNumber))
    time.sleep(2)

    # --- Step 4: Configure camera settings ---
    # Set white balance mode.
    COM_send_WLAN_command(SetWhiteBalanceMode())
    time.sleep(2)
    # Set camera properties (resolution, compression, etc.).
    COM_send_WLAN_command(SetCameraProperties())
    time.sleep(2)

    # --- Step 5: Start and perform a distance measurement to the measured object ---
    COM_send_WLAN_command(DoMeasure())
    time.sleep(4)

    # --- Step 6: Retrieve and process the measured distance ---
    # Get the measured distance to the object.
    distance_to_object = COM_send_WLAN_command(GetDistance()).split(",")[-1].strip()
    time.sleep(2)
    # Convert the distance to float and round to 4 decimal places.
    distance_to_object = round(float(distance_to_object), 4)
    print(f"Distance to object: {distance_to_object} m")

    if distance_to_object == 0.0:
        print("[ERROR] Distance to object is 0.0 m. Please start the process again or check the measurement.")
        sys.exit(1)

    # --- Step 7: Set the measured distance for the camera ---
    COM_send_WLAN_command(OVC_SetActDistance(distance_to_object))
    time.sleep(2)

    # --- Step 8: Capture an image with the OVC ---
    COM_send_WLAN_command(OVC_TakeImage(0))
    time.sleep(2)

    # --- Step 9: Abort any ongoing download process (precaution) ---
    COM_send_WLAN_command(AbortDownload())
    time.sleep(2)

    # --- Step 10: Download the captured image to the PC and delete it from the tachymeter ---
    downloadXL_and_delete_file(image_name)
    time.sleep(2)

    # --- Step 11: Abort any remaining download process (precaution) ---
    COM_send_WLAN_command(AbortDownload())
    time.sleep(2)

    # --- Step 12: Process the image to extract object corners automatically ---
    # This step includes segmentation, edge detection, Hough transform, Harris corner detection and DBSCAN-Clustering.
    # With the following function, two other scripts are called in sequence to perform the complete image processing workflow.
    # Make sure that the other scripts are correctly imported at the beginning of this script.
    image_processing_tachymeter(image_name)

    # --- Step 13: Read the detected object corners and bounding box offset from files ---
    corners_file = fr".\Harris_corner_detection_coordinates_{image_name}.txt"
    left_upper_corner_glob = fr".\left_upper_corner_BB_{image_name}.txt"
    pixelcoordinates = adjust_coordinates(left_upper_corner_glob, corners_file)
    print(pixelcoordinates)

    # --- Step 14: Get the crosshair position coordinates in the OCV-image ---
    response_cam_centre = COM_send_WLAN_command(OVC_GetCameraCentre()).split(":")[1]
    time.sleep(2)
    rdXCentre = round(float(response_cam_centre.split(",")[1].strip()))
    rdYCentre = round(float(response_cam_centre.split(",")[2].strip()))

    # --- Step 15: Move the telescope to each detected object corner and measure the cornerpoint ---
    if param == 0:
        for i, (pixx, pixy) in enumerate(pixelcoordinates):
            if i == 0:
                # Move to the first corner directly.
                print(f"[LOG] Turning telescope to pixel coordinates: X={pixx}, Y={pixy}")
                COM_send_WLAN_command(TelescopeToPixelCoord(pixx, pixy))
                time.sleep(2)

                # Turn the laser pointer on and off for visual confirmation.
                COM_send_WLAN_command(Laserpointer(1))
                time.sleep(2)
                COM_send_WLAN_command(Laserpointer(0))
                time.sleep(2)
            else:
                # Calculate the difference to the previous corner.
                prev_pixx, prev_pixy = pixelcoordinates[i - 1]
                diff_pixx = pixx - prev_pixx
                diff_pixy = pixy - prev_pixy
                print(f"Difference in pixel coordinates: X={diff_pixx}, Y={diff_pixy}")

                # Move to the next corner relative to the camera centre.
                new_pixx = rdXCentre + diff_pixx
                new_pixy = rdYCentre + diff_pixy
                print(f"[LOG] Turning telescope to pixel coordinates: X={new_pixx}, Y={new_pixy}")
                COM_send_WLAN_command(TelescopeToPixelCoord(new_pixx, new_pixy))
                time.sleep(2)

                # Turn the laser pointer on and off for visual confirmation.
                COM_send_WLAN_command(Laserpointer(1))
                time.sleep(2)
                COM_send_WLAN_command(Laserpointer(0))
                time.sleep(2)
    else:
        coordinates = []
        point_number = 1000
        for i, (pixx, pixy) in enumerate(pixelcoordinates):
            if i == 0:
                # Move to the first corner directly.
                print(f"[LOG] Turning telescope to pixel coordinates: X={pixx}, Y={pixy}")
                COM_send_WLAN_command(TelescopeToPixelCoord(pixx, pixy))
                time.sleep(2)

                # Start a distance measurement.
                COM_send_WLAN_command(DoMeasure())
                time.sleep(4)
                coord_output = COM_send_WLAN_command(GetCoordinate())
                time.sleep(1)
                # Stop the distance measurement.
                COM_send_WLAN_command(DoMeasure())
                time.sleep(2)
                try:
                    # Extract coordinates from the response and round them.
                    parts = coord_output.split(":")[-1].split(",")
                    east = round(float(parts[1]), 4)
                    north = round(float(parts[2]), 4)
                    height = round(float(parts[3]), 4)
                    # Save the coordinates in the format "point_number;east;north;height" in the list "coordinates".
                    coordinates.append(f"{point_number};{east};{north};{height}")
                    point_number += 1
                except Exception as e:
                    print(f"[ERROR] Could not extract coordinates from response: '{coord_output}'. Error: {e}")
            else:
                # Calculate the difference to the previous corner.
                prev_pixx, prev_pixy = pixelcoordinates[i - 1]
                diff_pixx = pixx - prev_pixx
                diff_pixy = pixy - prev_pixy
                print(f"Difference in pixel coordinates: X={diff_pixx}, Y={diff_pixy}")

                # Move to the next corner relative to the camera centre.
                new_pixx = rdXCentre + diff_pixx
                new_pixy = rdYCentre + diff_pixy
                print(f"[LOG] Turning telescope to pixel coordinates: X={new_pixx}, Y={new_pixy}")
                COM_send_WLAN_command(TelescopeToPixelCoord(new_pixx, new_pixy))
                time.sleep(2)

                # Start a distance measurement.
                COM_send_WLAN_command(DoMeasure())
                time.sleep(4)
                coord_output = COM_send_WLAN_command(GetCoordinate())
                time.sleep(1)
                # Stop the distance measurement.
                COM_send_WLAN_command(DoMeasure())
                time.sleep(2)
                try:
                    # Extract coordinates from the response and round them.
                    parts = coord_output.split(":")[-1].split(",")
                    east = round(float(parts[1]), 4)
                    north = round(float(parts[2]), 4)
                    height = round(float(parts[3]), 4)
                    # Save the coordinates in the format "point_number;east;north;height" in the list "coordinates".
                    coordinates.append(f"{point_number};{east};{north};{height}")
                    point_number += 1
                except Exception as e:
                    print(f"[ERROR] Could not extract coordinates from response: '{coord_output}'. Error: {e}")

        # Save the coordinates to a text file after each measurement
        with open("coordinates.txt", "w") as f:
            for line in coordinates:
                f.write(line + "\n")


# ================================
# 🔹 Main Function
# ================================
"""
The main function serves as the entry point for the automated measurement workflow.
It coordinates the overall process by calling the high-level workflow function,
which captures an image with the overview camera, processes the image, and controls
the tachymeter to measure the detected object corners. All commands are sent to the
tachymeter, and responses are logged for traceability.
"""

def main():
    COM_send_WLAN_command(Take_Image_with_OVC_and_turn_telescope_to_pixel_coordinates())
    

if __name__ == '__main__':
    main()