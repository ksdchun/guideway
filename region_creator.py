# ===============================================================================
# Region Creator Tool for Guideway
# ===============================================================================
# This application provides a graphical interface for creating and managing
# region of interest (ROI) definitions used by the Guideway object detection system.
# It allows users to define polygon regions over a webcam image that will be used
# for targeted object detection.
# ===============================================================================

import os 

# Disable hardware transforms in OpenCV to fix slow camera initialization issues
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

# GUI dependencies
import tkinter as tk                      # GUI framework
from tkinter import messagebox, simpledialog, ttk  # GUI dialogs and widgets
import sv_ttk                           # Sun Valley theme for modern UI

# Computer vision and image processing
import cv2                              # OpenCV for camera access
from PIL import Image, ImageTk, ImageDraw  # Image processing for overlays

# Data handling
import json                             # JSON parsing for region definitions
import copy                             # Deep copying for undo/redo functionality

# Path to regions JSON file - stored relative to script location
# This file contains all region definitions used by the main application
REGIONS_JSON_PATH = os.path.join(os.path.dirname(__file__), 'regions.json')

class RegionCreatorApp:
    """Main application class for the Region Creator tool.
    
    This class provides a graphical interface for creating and editing regions of interest
    that will be used by the Guideway object detection system. It allows users to:
    - Create and manage multiple region models (masks)
    - Draw polygon regions over a webcam image
    - Save regions to a JSON file for use by the main application
    - Undo/redo operations for easy editing
    """
    
    def push_undo(self):
        """Save the current state to the undo stack.
        
        This method creates a deep copy of the current data and selected model,
        adds it to the undo stack, and clears the redo stack (since a new action
        invalidates any previously undone actions).
        """
        # Create a deep copy of the current state and add to undo stack
        self.undo_stack.append((copy.deepcopy(self.data), self.model_var.get()))
        # Clear redo stack since we've taken a new action
        self.redo_stack.clear()
        # Update UI button states
        self._update_undo_redo_buttons()

    def __init__(self, root):
        """Initialize the Region Creator application.
        
        Sets up the main window, initializes the webcam, loads existing region data,
        and configures the user interface.
        
        Args:
            root: The Tkinter root window
        """
        # Store reference to root window
        self.root = root
        root.title("Region Creator")
        
        # Configure window to start maximized for best visibility
        root.state('zoomed')
        #root.resizable(False, False)  # Commented out to allow resizing
        
        # Apply Sun Valley theme for modern appearance
        sv_ttk.set_theme("light")  # Light theme for better visibility of regions
        
        # Initialize undo/redo functionality
        self.undo_stack = []  # Stack of previous states for undo
        self.redo_stack = []  # Stack of undone states for redo
        
        # Storage for semi-transparent fill overlays
        # These are kept as instance variables to prevent garbage collection
        self.region_fill_images = {}      # Overlay images for each region
        self.highlight_fill_image = None  # Overlay for highlighted region
        
        # Initialize data structure for region definitions
        self.data = {}  # Will hold all models and their regions
        self.load_json()  # Load existing regions if available
        
        # Initialize webcam - index 1 typically refers to the first external camera
        # (index 0 is usually the built-in webcam on laptops)
        self.cap = cv2.VideoCapture(1)
        
        # Set capture resolution to 1280x720 (720p HD)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Check if camera opened successfully
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Cannot open webcam")
            root.destroy()  # Close application if camera can't be accessed
            return
            
        # -----------------------------------------------------------------------
        # Camera warm-up period
        # -----------------------------------------------------------------------
        # Many webcams need time to adjust exposure and white balance
        # Capture several frames to allow the camera to auto-adjust before we
        # take the reference frame for region creation
        WARMUP_FRAMES = 20  # Number of frames to discard during warm-up
        
        print(f"[Info] Warming up camera ({WARMUP_FRAMES} frames)...")
        for i in range(WARMUP_FRAMES):
            ret, frame = self.cap.read()
            if not ret:
                messagebox.showerror("Error", "Cannot read from webcam")
                root.destroy()
                return
            # Short delay between frames to allow camera to adjust
            root.after(100)  # 100ms delay
            
        # Read one more frame now that camera has adjusted - this will be our reference frame
        print("[Info] Camera warm-up complete, capturing reference frame")
        ret, frame = self.cap.read()
        if not ret:
            messagebox.showerror("Error", "Cannot read from webcam")
            root.destroy()
            return
            
        # -----------------------------------------------------------------------
        # Prepare reference frame for region creation
        # -----------------------------------------------------------------------
        # Get dimensions from the captured frame
        self.frame_h, self.frame_w = frame.shape[:2]
        print(f"[Info] Using reference frame of size {self.frame_w}x{self.frame_h}")
        
        # Convert OpenCV BGR format to RGB for Tkinter compatibility
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Create PhotoImage for display in Tkinter canvas
        self.bg_image = ImageTk.PhotoImage(Image.fromarray(img))
        
        # -----------------------------------------------------------------------
        # Initialize drawing state variables
        # -----------------------------------------------------------------------
        self.start_x = self.start_y = None  # Starting point for region drawing
        self.current_points = []            # Points in current polygon being drawn
        self.point_markers = []             # Canvas IDs for point markers
        self.rect = None                    # Canvas ID for rectangle being drawn
        
        # -----------------------------------------------------------------------
        # Setup UI and keyboard shortcuts
        # -----------------------------------------------------------------------
        # Build the user interface
        self.setup_ui()
        
        # We use a static reference frame rather than live video
        # This makes it easier to precisely define regions
        # Uncomment to enable live video instead:
        # self.update_webcam()
        
        # Register keyboard shortcuts
        self.root.bind('<Control-z>', self._on_ctrl_z)  # Undo
        self.root.bind('<Control-y>', self._on_ctrl_y)  # Redo
        self.root.bind_all('n', self._on_n)             # New region
        self.root.bind_all('N', self._on_n)             # New region (uppercase)
        self.root.bind_all('<Delete>', self._on_delete)  # Delete selected region
        
        # Load initial model if one exists
        if self.model_var.get():
            self.load_model()  # Load regions for the selected model

    def load_json(self):
        """Load region definitions from the JSON file.
        
        Attempts to load existing region definitions from the regions.json file.
        If the file doesn't exist or can't be parsed, initializes with an empty
        data structure and shows a warning message.
        
        This method is crucial for the configuration persistence feature of the
        application, allowing region definitions to be saved and reused across
        sessions and by the main YOLO detection application.
        """
        # -----------------------------------------------------------------------
        # Attempt to load existing region definitions
        # -----------------------------------------------------------------------
        try:
            # Check if the regions file exists
            if os.path.exists(REGIONS_JSON_PATH):
                # Open and parse the JSON file
                with open(REGIONS_JSON_PATH, 'r') as f:
                    self.data = json.load(f)
                print(f"[Info] Successfully loaded regions from {REGIONS_JSON_PATH}")
                print(f"[Info] Found {len(self.data)} mask models")
            else:
                # File doesn't exist, initialize with empty data
                self.data = {}
                messagebox.showwarning(
                    "Warning", 
                    f"No regions file found at {REGIONS_JSON_PATH}.\n"
                    f"A new file will be created when you save."
                )
                print(f"[Info] No regions file found at {REGIONS_JSON_PATH}")
        except Exception as e:
            # Handle any errors during loading
            self.data = {}  # Initialize with empty data
            messagebox.showerror("Error", f"Failed to load regions: {e}")
            print(f"[Error] Failed to load regions: {e}")
            print(f"[Info] Initialized with empty data structure")

    def save_json(self):
        """Save region definitions to the JSON file.
        
        Writes the current region data to the regions.json file with proper
        formatting (indentation). This file will be used by the main application
        to define regions of interest for object detection.
        
        This method implements the configuration persistence feature that allows
        the YOLO detection system to use predefined regions for targeted detection,
        which is essential for the ROI masking functionality in the main application.
        """
        # -----------------------------------------------------------------------
        # Save current region definitions to JSON file
        # -----------------------------------------------------------------------
        try:
            # Create parent directory if it doesn't exist
            os.makedirs(os.path.dirname(REGIONS_JSON_PATH), exist_ok=True)
            
            # Write data to JSON file with nice formatting
            with open(REGIONS_JSON_PATH, 'w') as f:
                json.dump(self.data, f, indent=4)  # Use 4-space indentation for readability
                
            # Log success message
            print(f"[Info] Saved region definitions to {REGIONS_JSON_PATH}")
            print(f"[Info] Saved {len(self.data)} mask models")
            
            # Count total regions across all models
            total_regions = sum(len(model['regions']) for model in self.data.values())
            print(f"[Info] Total regions saved: {total_regions}")
            
        except Exception as e:
            # Handle any errors during saving
            messagebox.showerror("Error", f"Failed to save regions: {e}")
            print(f"[Error] Failed to save regions: {e}")
            print(f"[Error] Exception details: {type(e).__name__}")
            
            # Suggest recovery options
            print(f"[Info] Try saving to a different location or check file permissions")

    def setup_ui(self):
        # Fixed-width control panel on left
        ctrl = tk.Frame(self.root, width=100)
        ctrl.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        ctrl.pack_propagate(False)
        # Canvas click binding for region selection
        # (Canvas created later, so we'll bind after creation in this method)
        self.canvas_click_binding_set = False
        # Mask Name selection
        ttk.Label(ctrl, text="Mask Name:").pack()
        models = list(self.data.keys())
        if models:
            self.model_var = tk.StringVar(value=models[0])
            self.model_var.trace_add('write', lambda *args: self.load_model())
            self.model_menu = ttk.Combobox(ctrl, textvariable=self.model_var, values=models, state="readonly")
        else:
            self.model_var = tk.StringVar(value='')
            self.model_var.trace_add('write', lambda *args: self.load_model())
            self.model_menu = ttk.Combobox(ctrl, textvariable=self.model_var, values=[], state="readonly")
        self.model_menu.pack(fill=tk.X, pady=(0,8))
        ttk.Button(ctrl, text="New Mask...", command=self.add_model).pack(fill=tk.X, pady=2)
        ttk.Button(ctrl, text="Rename Mask", command=self.rename_model).pack(fill=tk.X, pady=2)
        ttk.Button(ctrl, text="Delete Mask", command=self.delete_model).pack(fill=tk.X, pady=2)
        # Regions list
        ttk.Label(ctrl, text="Region:").pack(pady=(10,0))
        self.region_listbox = tk.Listbox(ctrl, height=6)
        self.region_listbox.pack(fill=tk.X, pady=2)
        self.region_listbox.bind('<<ListboxSelect>>', self.on_region_select)
        # Region info panel
        self.info_label = ttk.Label(ctrl, text="Region Info:", anchor='w', justify='left')
        self.info_label.pack(fill=tk.X, pady=(10,0))
        # Detailed region info display
        self.region_info = tk.Text(ctrl, height=5, wrap='word', relief='solid', borderwidth=1)
        self.region_info.config(state='disabled')
        self.region_info.pack(fill=tk.X, pady=(0,10))
        # Undo/Redo buttons
        self.undo_btn = ttk.Button(ctrl, text="Undo", command=self.undo, state=tk.DISABLED)
        self.undo_btn.pack(fill=tk.X, pady=2)
        #self.undo_btn.tooltip = self._add_tooltip(self.undo_btn, "Undo (Ctrl+Z)")
        self.redo_btn = ttk.Button(ctrl, text="Redo", command=self.redo, state=tk.DISABLED)
        self.redo_btn.pack(fill=tk.X, pady=2)
        #self.redo_btn.tooltip = self._add_tooltip(self.redo_btn, "Redo (Ctrl+Y)")
        self.add_btn = ttk.Button(ctrl, text="Add Region", command=self.enable_draw)
        self.add_btn.pack(fill=tk.X, pady=2)
        #self._add_tooltip(self.add_btn, "Draw a new region on the canvas. (Shortcut: n)")
        self.del_btn = ttk.Button(ctrl, text="Delete Region", command=self.delete_region)
        self.del_btn.pack(fill=tk.X, pady=2)
        #self._add_tooltip(self.del_btn, "Delete the selected region.")
        # Button to finalize freeform region
        self.finish_btn = ttk.Button(ctrl, text="Finish Region", command=self.finish_polygon, state=tk.DISABLED)
        self.finish_btn.pack(fill=tk.X, pady=2)
        # Save/Clear
        self.clear_btn = ttk.Button(ctrl, text="Clear All Regions", command=self.clear_all_regions)
        self.clear_btn.pack(fill=tk.X, pady=(2,10))
        #self._add_tooltip(self.clear_btn, "Remove all regions for the selected pump model.")
        # Add Save and Close button
        ttk.Button(ctrl, text="Save and Close", command=self.save_and_close).pack(fill=tk.X, pady=(20,2))
        # Canvas for drawing
        self.canvas = tk.Canvas(self.root, width=self.frame_w, height=self.frame_h, cursor="cross")
        self.canvas.pack(side=tk.RIGHT)
        self.canvas.create_image(0, 0, image=self.bg_image, anchor='nw')
        if not self.canvas_click_binding_set:
            self.canvas.bind('<Button-1>', self.on_canvas_click)
            self.canvas_click_binding_set = True
        # Ensure model-dependent buttons are disabled if no model
        self._update_model_dependent_buttons()

    def add_model(self):
        """Add a new mask model.
        
        Prompts the user for a name and creates a new mask model with that name.
        Each model can contain multiple region definitions. The new model becomes
        the active selection after creation.
        """
        # Save current state for undo
        self.push_undo()
        
        # Prompt user for model name
        name = simpledialog.askstring("New Model", "Enter mask name:")
        if not name:
            # User cancelled or entered empty name
            return
            
        # Check if model already exists
        if name in self.data:
            messagebox.showwarning("Warning", f"Model '{name}' already exists.")
            return
            
        # Create new model with empty regions dictionary
        self.data[name] = {"regions": {}}
        print(f"[Info] Created new mask model: {name}")
        
        # Update UI with new model
        models = list(self.data.keys())
        self.model_menu['values'] = models  # Update dropdown values
        self.model_var.set(name)           # Select the new model
        
        # Save changes and refresh UI
        self.save_json()
        self.load_model()  # Load (empty) regions for the new model

    def _update_model_dependent_buttons(self):
        """Update button states based on current model selection.
        
        Enables or disables buttons depending on whether a valid model is selected
        and whether the model has any regions defined.
        """
        # Get current model selection
        model = self.model_var.get()
        
        # Check if we have a valid model selected
        valid = bool(model and model in self.data)
        
        # Check if the model has any regions
        regions_exist = False
        if valid:
            regions_exist = bool(self.data[model]['regions'])
            
        # Enable "Add Region" button only if we have a valid model
        self.add_btn.config(state=(tk.NORMAL if valid else tk.DISABLED))
        
        # Enable "Delete Region" button only if there are regions to delete
        self.del_btn.config(state=(tk.NORMAL if regions_exist else tk.DISABLED))

    def load_model(self):
        """Load and display the regions for the currently selected model.
        
        Clears any existing regions from the canvas and loads the regions
        for the currently selected model. Updates the region listbox and
        creates visual representations of each region on the canvas.
        """
        # Get current model selection
        model = self.model_var.get()
        
        # -----------------------------------------------------------------------
        # Clear existing regions from display
        # -----------------------------------------------------------------------
        self.canvas.delete("region")    # Delete all region polygons
        self.canvas.delete("highlight") # Delete any highlighted region
        self.region_fill_images = {}    # Clear overlay images
        self.highlight_fill_image = None
        
        # -----------------------------------------------------------------------
        # Update region listbox
        # -----------------------------------------------------------------------
        self.region_listbox.delete(0, tk.END)  # Clear listbox
        
        # Exit if no valid model is selected
        if not model or model not in self.data:
            print("[Info] No valid model selected or model doesn't exist")
            return
            
        # -----------------------------------------------------------------------
        # Add regions to listbox and canvas
        # -----------------------------------------------------------------------
        regions = self.data[model]['regions']
        print(f"[Info] Loading model '{model}' with {len(regions)} regions")
        
        # Add region names to listbox
        for region_name in regions:
            self.region_listbox.insert(tk.END, region_name)
            
        # Draw each region on the canvas
        for region_name, points in regions.items():
            # Draw polygon outline in blue
            coords = [coord for point in points for coord in point]  # Flatten points list
            self.canvas.create_polygon(
                coords,
                outline='blue',
                fill='',       # No fill in the polygon itself
                width=2,
                tags=("region", region_name)  # Tag with both "region" and the region name
            )
            
            # Create semi-transparent fill overlay for this region
            overlay = Image.new('RGBA', (self.frame_w, self.frame_h), (0, 0, 0, 0))  # Transparent
            draw = ImageDraw.Draw(overlay)
            draw.polygon(points, fill=(0, 0, 255, 64))  # Semi-transparent blue
            self.region_fill_images[region_name] = ImageTk.PhotoImage(overlay)
            
        # -----------------------------------------------------------------------
        # Update button states based on new model
        # -----------------------------------------------------------------------
        self._update_model_dependent_buttons()

    def setup_ui(self):
        """Set up the user interface components.
        
        Creates and configures all UI elements including:
        - Left control panel with model and region controls
        - Canvas for displaying the webcam image and drawing regions
        - Buttons for various operations (add, delete, undo, etc.)
        """
        # -----------------------------------------------------------------------
        # Control Panel Setup (Left Side)
        # -----------------------------------------------------------------------
        # Create fixed-width control panel on left side of window
        ctrl = tk.Frame(self.root, width=140)
        ctrl.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        ctrl.pack_propagate(False)  # Prevent control panel from resizing
        
        # Initialize canvas click binding flag
        # (Canvas created later, so we'll bind after creation in this method)
        self.canvas_click_binding_set = False
        
        # -----------------------------------------------------------------------
        # Mask Model Selection Section
        # -----------------------------------------------------------------------
        ttk.Label(ctrl, text="Mask Name:").pack()
        
        # Get list of available models from loaded data
        models = list(self.data.keys())
        
        # Create model selection dropdown with appropriate initial value
        if models:
            # If models exist, select the first one
            self.model_var = tk.StringVar(value=models[0])
            self.model_var.trace_add('write', lambda *args: self.load_model())  # Auto-load when changed
            self.model_menu = ttk.Combobox(ctrl, textvariable=self.model_var, values=models, state="readonly")
        else:
            # If no models exist, create empty dropdown
            self.model_var = tk.StringVar(value='')
            self.model_var.trace_add('write', lambda *args: self.load_model())
            self.model_menu = ttk.Combobox(ctrl, textvariable=self.model_var, values=[], state="readonly")
            
        self.model_menu.pack(fill=tk.X, pady=(0,8))
        
        # -----------------------------------------------------------------------
        # Model Management Buttons
        # -----------------------------------------------------------------------
        ttk.Button(ctrl, text="New Mask...", command=self.add_model).pack(fill=tk.X, pady=2)
        ttk.Button(ctrl, text="Rename Mask", command=self.rename_model).pack(fill=tk.X, pady=2)
        ttk.Button(ctrl, text="Delete Mask", command=self.delete_model).pack(fill=tk.X, pady=2)
        # -----------------------------------------------------------------------
        # Region List and Selection
        # -----------------------------------------------------------------------
        ttk.Label(ctrl, text="Region:").pack(pady=(10,0))
        
        # Listbox for displaying and selecting regions
        self.region_listbox = tk.Listbox(ctrl, height=6)
        self.region_listbox.pack(fill=tk.X, pady=2)
        self.region_listbox.bind('<<ListboxSelect>>', self.on_region_select)  # Bind selection event
        
        # -----------------------------------------------------------------------
        # Region Information Display
        # -----------------------------------------------------------------------
        self.info_label = ttk.Label(ctrl, text="Region Info:", anchor='w', justify='left')
        self.info_label.pack(fill=tk.X, pady=(10,0))
        
        # Text widget for displaying detailed region information
        self.region_info = tk.Text(ctrl, height=5, wrap='word', relief='solid', borderwidth=1)
        self.region_info.config(state='disabled')  # Read-only initially
        self.region_info.pack(fill=tk.X, pady=(0,10))
        
        # -----------------------------------------------------------------------
        # Editing Controls
        # -----------------------------------------------------------------------
        # Undo/Redo functionality
        self.undo_btn = ttk.Button(ctrl, text="Undo", command=self.undo, state=tk.DISABLED)
        self.undo_btn.pack(fill=tk.X, pady=2)
        self.redo_btn = ttk.Button(ctrl, text="Redo", command=self.redo, state=tk.DISABLED)
        self.redo_btn.pack(fill=tk.X, pady=2)
        
        # Region creation and deletion
        self.add_btn = ttk.Button(ctrl, text="Add Region", command=self.enable_draw)
        self.add_btn.pack(fill=tk.X, pady=2)
        self.del_btn = ttk.Button(ctrl, text="Delete Region", command=self.delete_region)
        self.del_btn.pack(fill=tk.X, pady=2)
        
        # Button to finalize polygon drawing (only enabled during drawing)
        self.finish_btn = ttk.Button(ctrl, text="Finish Region", command=self.finish_polygon, state=tk.DISABLED)
        self.finish_btn.pack(fill=tk.X, pady=2)
        # Add Save and Close button
        ttk.Button(ctrl, text="Save and Close", command=self.save_and_close).pack(fill=tk.X, pady=(20,2))
        # Canvas for drawing
        self.canvas = tk.Canvas(self.root, width=self.frame_w, height=self.frame_h, cursor="cross")
        self.canvas.pack(side=tk.RIGHT)
        self.canvas.create_image(0, 0, image=self.bg_image, anchor='nw')
        if not self.canvas_click_binding_set:
            self.canvas.bind('<Button-1>', self.on_canvas_click)
            self.canvas_click_binding_set = True
        # Ensure model-dependent buttons are disabled if no model
        self._update_model_dependent_buttons()

    def cancel_draw_mode(self):
        """Cancel the current region drawing operation.
        
        Resets the canvas to its normal state by:
        - Removing event bindings for drawing
        - Clearing any in-progress points or markers
        - Restoring the normal click behavior
        - Disabling the finish button
        
        This is called when drawing is canceled or completed.
        """
        # -----------------------------------------------------------------------
        # Remove drawing-related event bindings
        # -----------------------------------------------------------------------
        self.canvas.unbind("<ButtonPress-1>")  # Mouse press
        self.canvas.unbind("<B1-Motion>")      # Mouse drag
        self.canvas.unbind("<ButtonRelease-1>")  # Mouse release
        
        # -----------------------------------------------------------------------
        # Clear any in-progress drawing elements
        # -----------------------------------------------------------------------
        # Remove any point markers that have been placed
        for marker in getattr(self, 'point_markers', []):
            self.canvas.delete(marker)  # Remove marker from canvas
        
        # Clear point collections
        self.point_markers.clear()  # Canvas marker IDs
        self.current_points.clear()  # Coordinate points
        
        # -----------------------------------------------------------------------
        # Restore normal canvas behavior
        # -----------------------------------------------------------------------
        # Restore normal click behavior for region selection
        self.canvas.bind('<Button-1>', self.on_canvas_click)
        
        # Remove any rectangle being drawn
        if self.rect:
            self.canvas.delete(self.rect)
            self.rect = None
            
        # Disable the finish button since we're no longer drawing
        self.finish_btn.config(state=tk.DISABLED)

    def rename_model(self):
        """Rename the currently selected mask model.
        
        Prompts the user for a new name and updates the model key in the data
        dictionary. Validates that a model is selected and that the new name
        doesn't conflict with an existing model.
        """
        # Save current state for undo
        self.push_undo()
        
        # Get current model selection
        model = self.model_var.get()
        
        # Validate that a model is selected
        if not model or model not in self.data:
            messagebox.showwarning("Warning", "Select a valid model first.")
            return
            
        # Prompt for new name with current name as default
        new_name = simpledialog.askstring(
            "Rename Model", 
            f"Enter new name for model '{model}':", 
            initialvalue=model
        )
        
        # Exit if canceled or name unchanged
        if not new_name or new_name == model:
            return
            
        # Check if new name already exists
        if new_name in self.data:
            messagebox.showwarning("Warning", f"Model '{new_name}' already exists.")
            return
            
        # Rename the model by moving data to new key
        self.data[new_name] = self.data.pop(model)  # Move data to new key
        print(f"[Info] Renamed model '{model}' to '{new_name}'")
        
        # Save changes
        self.save_json()
        
        # Update UI
        models = list(self.data.keys())
        self.model_menu['values'] = models  # Update dropdown values
        self.model_var.set(new_name)        # Select the renamed model
        self.load_model()                   # Reload regions for the model

    def delete_model(self):
        """Delete the currently selected mask model.
        
        Removes the selected model and all its regions from the data dictionary
        after confirmation. Updates the UI to reflect the change and selects
        another model if available.
        """
        # Save current state for undo
        self.push_undo()
        
        # Cancel any active drawing operation
        self.cancel_draw_mode()
        
        # Get current model selection
        model = self.model_var.get()
        
        # Validate that a model is selected
        if not model or model not in self.data:
            messagebox.showwarning("Warning", "Select a valid model first.")
            return
            
        # Confirm deletion with user
        if not messagebox.askyesno(
            "Confirm", 
            f"Delete model '{model}'?\nThis will remove all regions for this model."
        ):
            return  # User canceled
            
        # Delete the model from data dictionary
        del self.data[model]
        print(f"[Info] Deleted model '{model}'")
        
        # Save changes
        self.save_json()
        
        # Update UI
        models = list(self.data.keys())
        self.model_menu['values'] = models  # Update dropdown values
        
        # Select another model if available, otherwise clear selection
        if models:
            self.model_var.set(models[0])  # Select first available model
        else:
            self.model_var.set('')         # No models left
            
        # Reload the view
        self.load_model()

    def on_region_select(self, event):
        """Handle region selection from the listbox.
        
        When a region is selected in the listbox, this method:
        1. Highlights the region on the canvas
        2. Displays detailed information about the region
        3. Creates a semi-transparent overlay to visually identify the region
        
        Args:
            event: The listbox selection event
        """
        # Exit if no selection was made
        if not event.widget.curselection():
            # Clear region info if nothing selected
            self.region_info.config(state='normal')
            self.region_info.delete(1.0, tk.END)
            self.region_info.config(state='disabled')
            return
            
        # Get the selected region name from the listbox
        idx = event.widget.curselection()[0]
        region_name = event.widget.get(idx)
        
        # Get current model and validate region exists
        model = self.model_var.get()
        if not model or model not in self.data or region_name not in self.data[model]['regions']:
            # Clear region info if region doesn't exist
            self.region_info.config(state='normal')
            self.region_info.delete(1.0, tk.END)
            self.region_info.config(state='disabled')
            return
            
        # -----------------------------------------------------------------------
        # Highlight the selected region on the canvas
        # -----------------------------------------------------------------------
        # Remove any existing highlight
        self.canvas.delete("highlight")
        
        # Get region points from data
        points = self.data[model]['regions'][region_name]
        
        # Draw yellow outline around the region
        coords = [c for p in points for c in p]  # Flatten points list
        self.canvas.create_polygon(
            *coords, 
            outline="yellow",  # Bright yellow outline
            width=3,          # Thicker line for visibility
            fill="",          # No fill (transparent)
            tags=("highlight", region_name)  # Tag for easy identification
        )
        
        # Create semi-transparent yellow highlight overlay
        overlay = Image.new("RGBA", (self.frame_w, self.frame_h), (0, 0, 0, 0))  # Transparent base
        draw = ImageDraw.Draw(overlay)
        draw.polygon(coords, fill=(255, 255, 0, 80))  # Semi-transparent yellow
        
        # Create and display the highlight overlay
        self.highlight_fill_image = ImageTk.PhotoImage(overlay)
        self.canvas.create_image(
            0, 0, 
            image=self.highlight_fill_image, 
            anchor="nw", 
            tags=("highlight", f"{region_name}_fill")
        )
        
        # -----------------------------------------------------------------------
        # Update region information display
        # -----------------------------------------------------------------------
        # Enable text widget for editing
        self.region_info.config(state='normal')
        self.region_info.delete(1.0, tk.END)  # Clear existing content
        
        # Add region details
        self.region_info.insert(tk.END, f"Region: {region_name}\n")
        self.region_info.insert(tk.END, f"Points: {len(points)}\n")
        self.region_info.insert(tk.END, f"Coordinates:\n")
        
        # List all points in the region
        for i, pt in enumerate(points):
            self.region_info.insert(tk.END, f"  {i+1}: ({pt[0]}, {pt[1]})\n")
            
        # Disable text widget to make it read-only
        self.region_info.config(state='disabled')
        
        print(f"[Info] Selected region: {region_name}")

    def on_canvas_click(self, event):
        """Handle clicks on the canvas for region selection.
        
        This method detects when a user clicks on a region in the canvas and
        selects that region in the listbox. It uses canvas tags to identify
        which region was clicked.
        
        Args:
            event: The canvas click event containing x,y coordinates
        """
        # Skip if we're in drawing mode (handled by other methods)
        if hasattr(self, 'drawing_mode') and self.drawing_mode:
            return
            
        # Validate current model
        model = self.model_var.get()
        if not model or model not in self.data:
            return
            
        # -----------------------------------------------------------------------
        # Find region under the click point
        # -----------------------------------------------------------------------
        # Get all items with the "region" tag
        items = self.canvas.find_withtag("region")
        
        # Check each region to see if it was clicked
        for item in items:
            # Only process polygon items
            if self.canvas.type(item) == "polygon":
                # Check if click point is within or very near the polygon
                # Create a small bounding box around the click point
                if self.canvas.find_overlapping(event.x-2, event.y-2, event.x+2, event.y+2):
                    # Get all tags for this item
                    tags = self.canvas.gettags(item)
                    
                    # Look for a tag that matches a region name
                    for tag in tags:
                        if tag != "region" and tag in self.data[model]['regions']:
                            # -----------------------------------------------
                            # Select this region in the listbox
                            # -----------------------------------------------
                            for i in range(self.region_listbox.size()):
                                if self.region_listbox.get(i) == tag:
                                    # Clear any existing selection
                                    self.region_listbox.selection_clear(0, tk.END)
                                    # Select this region
                                    self.region_listbox.selection_set(i)
                                    # Ensure the selection is visible
                                    self.region_listbox.see(i)
                                    # Trigger the selection handler
                                    self.on_region_select(event)
                                    
                                    print(f"[Info] Clicked on region: {tag}")
                                    return

    def delete_region(self):
        """Delete the currently selected region.
        
        Removes the selected region from the current model after validating
        that both a model and region are selected. Updates the UI to reflect
        the change.
        """
        # Save current state for undo
        self.push_undo()
        
        # Get current model and validate
        model = self.model_var.get()
        if not model or model not in self.data:
            messagebox.showwarning("Warning", "Select a valid model first.")
            return
            
        # Get selected region and validate
        selection = self.region_listbox.curselection()
        if not selection:
            messagebox.showwarning("Warning", "Select a region first.")
            return
            
        # Get region name and remove from data
        region = self.region_listbox.get(selection[0])
        del self.data[model]['regions'][region]
        print(f"[Info] Deleted region: {region} from model: {model}")
        
        # Always clear highlights before reloading the model
        self.canvas.delete('highlight')
        self.canvas.delete('highlight_fill')
        
        # Clear the region info text
        self.region_info.config(state='normal')
        self.region_info.delete('1.0', tk.END)
        self.region_info.config(state='disabled')
        
        # Save changes and reload the model
        self.save_json()
        self.load_model()

    def enable_draw(self):
        """Enable drawing mode for creating a new region.
        
        Switches the application into drawing mode where the user can click
        on the canvas to place points defining a polygon region. Validates
        that a model is selected before enabling drawing.
        """
        # Save current state for undo
        self.push_undo()
        
        # Validate that a model is selected
        model = self.model_var.get()
        if not model:
            messagebox.showwarning("Warning", "Select or create a model first.")
            return
            
        # -----------------------------------------------------------------------
        # Prepare canvas for drawing mode
        # -----------------------------------------------------------------------
        # Remove normal click behavior for region selection
        self.canvas.unbind('<Button-1>')
        
        # Clear any existing point markers
        for marker in self.point_markers:
            self.canvas.delete(marker)
            
        # Reset point collections
        self.point_markers = []    # Canvas marker IDs
        self.current_points = []   # Coordinate points
        
        # -----------------------------------------------------------------------
        # Set up drawing interaction
        # -----------------------------------------------------------------------
        # Bind click event to point placement function
        self.canvas.bind("<Button-1>", self.on_point_click)
        
        # Enable the finish button to allow completing the polygon
        self.finish_btn.config(state=tk.NORMAL)
        
        print(f"[Info] Entered drawing mode for model: {model}")

    def on_point_click(self, event):
        """Handle clicks when in drawing mode to create polygon points.
        
        Each click adds a point to the current polygon being drawn. Points are
        visualized with blue markers on the canvas.
        
        Args:
            event: The canvas click event containing x,y coordinates
        """
        # Get click coordinates
        x, y = event.x, event.y
        
        # Record point in the current polygon
        self.current_points.append((x, y))
        
        # Draw visual marker at the clicked point
        r = 5  # Radius of marker circle
        marker = self.canvas.create_oval(
            x-r, y-r, x+r, y+r,  # Create circle with 10px diameter
            fill="blue",          # Solid blue fill
            tags="point_marker"   # Tag for identification
        )
        self.point_markers.append(marker)  # Store marker ID for later removal

    def finish_polygon(self):
        """Complete the polygon region being drawn and save it.
        
        This method is called when the user clicks the 'Finish Region' button.
        It validates that enough points have been placed, processes the points to
        create a valid polygon, and saves the region to the current model.
        
        The method also ensures the polygon is non-self-intersecting by sorting
        points radially around their centroid.
        """
        # -----------------------------------------------------------------------
        # Validate polygon has enough points
        # -----------------------------------------------------------------------
        # Need at least 3 points to form a valid polygon
        if len(self.current_points) < 3:
            messagebox.showwarning("Warning", "Need at least 3 points to finish a region.")
            return
            
        # Validate model is still selected
        model = self.model_var.get()
        if not model or model not in self.data:
            return
            
        # -----------------------------------------------------------------------
        # Process points to create a valid polygon
        # -----------------------------------------------------------------------
        # Clamp points to ensure they're within frame boundaries
        pts = [
            (max(0, min(x, self.frame_w-1)), max(0, min(y, self.frame_h-1))) 
            for x, y in self.current_points
        ]
        
        # Sort points radially around centroid to create non-self-intersecting polygon
        import math
        # Calculate centroid (average of all points)
        cx = sum(x for x, y in pts) / len(pts)

    def _on_ctrl_z(self, event=None):
        """Handle Ctrl+Z keyboard shortcut for undo operation.
        
        Args:
            event: The keyboard event (optional)
            
        Returns:
            "break" to prevent default event handling
        """
        self.undo()
        return "break"  # Prevent default handling

    def _on_ctrl_y(self, event=None):
        """Handle Ctrl+Y keyboard shortcut for redo operation.
        
        Args:
            event: The keyboard event (optional)
            
        Returns:
            "break" to prevent default event handling
        """
        self.redo()
        return "break"  # Prevent default handling

    def _on_n(self, event=None):
        """Handle 'N' keyboard shortcut for creating a new region.
        
        Only activates if the Add Region button is enabled.
        
        Args:
            event: The keyboard event (optional)
            
        Returns:
            "break" to prevent default event handling
        """
        if self.add_btn.instate(['!disabled']):  # Check if button is not disabled
            self.enable_draw()  # Enter drawing mode
        return "break"  # Prevent default handling

    def _on_delete(self, event=None):
        """Handle Delete key for removing the selected region.
        
        Only activates if the Delete Region button is enabled and a region is selected.
        
        Args:
            event: The keyboard event (optional)
            
        Returns:
            "break" to prevent default event handling
        """
        # Check if delete button is enabled and a region is selected
        if self.del_btn.instate(['!disabled']) and self.region_listbox.curselection():
            self.delete_region()  # Delete the selected region
        return "break"  # Prevent default handling

    def undo(self):
        """Restore the previous state from the undo stack.
        
        This method implements undo functionality by:
        1. Saving the current state to the redo stack
        2. Restoring the previous state from the undo stack
        3. Updating the UI to reflect the restored state
        
        Does nothing if the undo stack is empty.
        """
        # Check if there's anything to undo
        if not self.undo_stack:
            return
            
        # Import copy module for deep copying objects
        import copy
        
        # Save current state to redo stack before undoing
        self.redo_stack.append((copy.deepcopy(self.data), self.model_var.get()))
        
        # Restore previous state from undo stack
        self.data, model = self.undo_stack.pop()
        
        # Update UI to match restored state
        self.model_var.set(model)  # Set model selection
        self.save_json()           # Save changes to disk
        self.load_model()          # Reload UI with restored data
        
        # Update undo/redo button states
        self._update_undo_redo_buttons()
        
        print(f"[Info] Undo operation completed, restored model: {model}")

    def redo(self):
        """Restore a previously undone state from the redo stack.
        
        This method implements redo functionality by:
        1. Saving the current state to the undo stack
        2. Restoring the next state from the redo stack
        3. Updating the UI to reflect the restored state
        
        Does nothing if the redo stack is empty.
        """
        # Check if there's anything to redo
        if not self.redo_stack:
            return
            
        # Import copy module for deep copying objects
        import copy
        
        # Save current state to undo stack before redoing
        self.undo_stack.append((copy.deepcopy(self.data), self.model_var.get()))
        
        # Restore next state from redo stack
        self.data, model = self.redo_stack.pop()
        
        # Update UI to match restored state
        self.model_var.set(model)  # Set model selection
        self.save_json()           # Save changes to disk
        self.load_model()          # Reload UI with restored data
        
        # Update undo/redo button states
        self._update_undo_redo_buttons()
        
        print(f"[Info] Redo operation completed, restored model: {model}")

    def on_motion(self, event):
        """Handle mouse movement during rectangle drawing.
        
        Updates the size and position of the rectangle being drawn as the user
        moves the mouse. Ensures the rectangle stays within canvas boundaries.
        
        Args:
            event: The mouse motion event containing current x,y coordinates
        """
        # Define margin to keep rectangle within visible bounds
        MARGIN = 4
        
        # -----------------------------------------------------------------------
        # Clamp coordinates to ensure they stay within canvas boundaries
        # -----------------------------------------------------------------------
        # Clamp starting point (where mouse was first pressed)
        x1 = min(max(self.start_x, MARGIN), self.frame_w - 1 - MARGIN)
        y1 = min(max(self.start_y, MARGIN), self.frame_h - 1 - MARGIN)
        
        # Clamp current point (where mouse is now)
        x2 = min(max(event.x, MARGIN), self.frame_w - 1)
        y2 = min(max(event.y, MARGIN), self.frame_h - 1)
        
        # -----------------------------------------------------------------------
        # Calculate rectangle coordinates (top-left to bottom-right)
        # -----------------------------------------------------------------------
        # Find leftmost x coordinate
        left = min(x1, x2)
        # Find rightmost x coordinate
        right = max(x1, x2)
        # Find topmost y coordinate
        top = min(y1, y2)
        # Find bottommost y coordinate
        bottom = max(y1, y2)
        
        # Update the rectangle's position and size on the canvas
        self.canvas.coords(self.rect, left, top, right, bottom)

    def on_release(self, event):
        """Handle mouse release to complete rectangle drawing.
        
        When the user releases the mouse button, this method finalizes the
        rectangle region by:
        1. Validating the model selection
        2. Calculating the final rectangle coordinates
        3. Creating a polygon representation of the rectangle
        
        Args:
            event: The mouse release event containing final x,y coordinates
        """
        # -----------------------------------------------------------------------
        # Validate model selection
        # -----------------------------------------------------------------------
        # Check that model is still valid before adding region
        model = self.model_var.get()
        if not model or model not in self.data:
            self.cancel_draw_mode()  # Cancel drawing if model is invalid
            return
            
        # Save current state for undo
        self.push_undo()
        
        # -----------------------------------------------------------------------
        # Calculate final rectangle coordinates
        # -----------------------------------------------------------------------
        # Get starting point (where mouse was first pressed)
        x1, y1 = self.start_x, self.start_y
        
        # Get ending point (where mouse was released), clamped to canvas boundaries
        x2 = min(max(event.x, 0), self.frame_w - 1)
        y2 = min(max(event.y, 0), self.frame_h - 1)
        
        # Calculate rectangle corners
        left = min(x1, x2)    # Leftmost x coordinate
        right = max(x1, x2)   # Rightmost x coordinate
        top = min(y1, y2)     # Topmost y coordinate
        bottom = max(y1, y2)  # Bottommost y coordinate
        
        # Create polygon representation of rectangle (4 corners in clockwise order)
        rect = [
            (left, top),      # Top-left corner
            (right, top),     # Top-right corner
            (right, bottom),  # Bottom-right corner
            (left, bottom)    # Bottom-left corner
        ]

        # -----------------------------------------------------------------------
        # Create and save the new region
        # -----------------------------------------------------------------------
        # Generate a unique region name by incrementing counter
        existing = self.data[model]['regions']
        next_num = 1
        while f"region-{next_num}" in existing:  # Find unused number
            next_num += 1
        region_name = f"region-{next_num}"  # Format: "region-1", "region-2", etc.
        
        # Add the new region to the model's regions
        self.data[model]['regions'][region_name] = rect
        print(f"[Info] Created new rectangle region: {region_name} at {rect}")
        
        # Save changes to disk
        self.save_json()
        
        # -----------------------------------------------------------------------
        # Clean up drawing state
        # -----------------------------------------------------------------------
        # Remove any temporary drawing elements
        self.canvas.delete("drawing")
        
        # -----------------------------------------------------------------------
        # Restore normal interaction mode
        # -----------------------------------------------------------------------
        # Remove drawing-related event bindings
        self.canvas.unbind("<ButtonPress-1>")  # Mouse press
        self.canvas.unbind("<B1-Motion>")      # Mouse drag
        self.canvas.unbind("<ButtonRelease-1>")  # Mouse release
        
        # Restore normal region selection click binding
        self.canvas.bind('<Button-1>', self.on_canvas_click)
        
        # Reload the model to display the new region
        self.load_model()
        
        # Update undo/redo button states
        self._update_undo_redo_buttons()

    def update_webcam(self):
        """Update the webcam feed displayed on the canvas.
        
        This method is called repeatedly to refresh the webcam image shown on
        the canvas. It captures a frame from the webcam, converts it to the
        proper format for display, and schedules itself to run again.
        
        The method runs at approximately 30 frames per second (every 33ms).
        """
        # Capture a frame from the webcam
        ret, frame = self.cap.read()
        
        # If frame was successfully captured
        if ret:
            # Convert from BGR (OpenCV format) to RGB (PIL format)
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Convert to a format Tkinter can display
            self.bg_image = ImageTk.PhotoImage(Image.fromarray(img))
            
            # Display the image on the canvas
            self.canvas.create_image(0, 0, image=self.bg_image, anchor='nw')
            
        # Schedule this method to run again in 33ms (~30 FPS)
        self.root.after(33, self.update_webcam)

    def on_close(self):
        """Handle application closing.
        
        This method is called when the user closes the application. It performs
        necessary cleanup tasks:
        1. Saves the current regions to the JSON file
        2. Displays a confirmation message to the user
        3. Releases the webcam resource
        4. Destroys the Tkinter window
        """
        # Save current regions to the JSON file
        self.save_json()
        
        # Show confirmation message to the user
        messagebox.showinfo("Saved", f"Region saved to {REGIONS_JSON_PATH}.")
        
        # Release the webcam to free system resources
        self.cap.release()
        
        # Close the application window
        self.root.destroy()
        
        print("[Info] Region Creator application closed")

def main():
    root = tk.Tk()
    app = RegionCreatorApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()

def save_and_close(self):
    self.save_json()
    messagebox.showinfo("Saved", f"Region saved to {REGIONS_JSON_PATH}.")
    self.root.destroy()

# Attach method to class
setattr(RegionCreatorApp, 'save_and_close', save_and_close)

if __name__ == '__main__':
    main()