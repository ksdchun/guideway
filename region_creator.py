import os 

os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0" # Fix for external camera took long time to load

import tkinter as tk
from tkinter import messagebox, simpledialog, ttk
import sv_ttk
import cv2
from PIL import Image, ImageTk, ImageDraw
import json
import copy

# Path to regions JSON file
REGIONS_JSON_PATH = os.path.join(os.path.dirname(__file__), 'regions.json')

class RegionCreatorApp:
    def push_undo(self):
        
        self.undo_stack.append((copy.deepcopy(self.data), self.model_var.get()))
        self.redo_stack.clear()
        self._update_undo_redo_buttons()

    def __init__(self, root):

        self.root = root
        root.title("Region Creator")
        # Start maximized
        root.state('zoomed')
        #root.resizable(False, False)
        # Apply Sun Valley ttk theme (dark)
        sv_ttk.set_theme("light")
        # Undo/Redo stacks
        self.undo_stack = []
        self.redo_stack = []
        # storage for semi-transparent fill overlays
        self.region_fill_images = {}
        self.highlight_fill_image = None
        # Load existing data
        self.data = {}
        self.load_json()
        # Initialize webcam
        self.cap = cv2.VideoCapture(1)
        # Set capture resolution to 1280x720
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Cannot open webcam")
            root.destroy()
            return
            
        # Add a camera warm-up period to allow for auto-adjustment (silently)
        # Capture several frames to allow the camera to adjust exposure
        WARMUP_FRAMES = 20
        for i in range(WARMUP_FRAMES):
            ret, frame = self.cap.read()
            if not ret:
                messagebox.showerror("Error", "Cannot read from webcam")
                root.destroy()
                return
            # Short delay between frames
            root.after(100)
            
        # Read one more frame now that camera has adjusted
        ret, frame = self.cap.read()
        if not ret:
            messagebox.showerror("Error", "Cannot read from webcam")
            root.destroy()
            return
            
        # Use captured frame resolution directly
        self.frame_h, self.frame_w = frame.shape[:2]
        # Convert to PhotoImage
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.bg_image = ImageTk.PhotoImage(Image.fromarray(img))
        # State
        self.start_x = self.start_y = None
        self.current_points = []
        self.point_markers = []
        self.rect = None
        # Build UI
        self.setup_ui()
        # Only use initial frame for drawing; disable live update
        # self.update_webcam()
        # Keyboard shortcuts for undo/redo, add region, and delete region
        self.root.bind('<Control-z>', self._on_ctrl_z)
        self.root.bind('<Control-y>', self._on_ctrl_y)
        self.root.bind_all('n', self._on_n)
        self.root.bind_all('N', self._on_n)
        self.root.bind_all('<Delete>', self._on_delete)
        # Load initial model (if any)
        if self.model_var.get():
            self.load_model()

    def load_json(self):
        if os.path.exists(REGIONS_JSON_PATH):
            try:
                with open(REGIONS_JSON_PATH, 'r') as f:
                    self.data = json.load(f)
            except Exception:
                messagebox.showwarning("Warning", "Failed to load regions.json. Starting fresh.")
                self.data = {}
        else:
            self.data = {}

    def save_json(self):
        with open(REGIONS_JSON_PATH, 'w') as f:
            json.dump(self.data, f, indent=2)
        #messagebox.showinfo("Saved", f"Regions saved to {REGIONS_JSON_PATH}.")

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
        # Bind canvas click for region selection
        # Ensure model-dependent buttons are disabled if no model
        self._update_model_dependent_buttons()

    def add_model(self):
        self.push_undo()
        name = simpledialog.askstring("New Model", "Enter mask name:")
        if not name:
            return
        if name in self.data:
            messagebox.showwarning("Warning", f"Model '{name}' already exists.")
            return
        self.data[name] = {"regions": {}}
        models = list(self.data.keys())
        self.model_menu['values'] = models
        self.model_var.set(name)
        self.save_json()
        self.load_model()

    def _update_model_dependent_buttons(self):
        model = self.model_var.get()
        valid = bool(model and model in self.data)
        regions_exist = False
        if valid:
            regions_exist = bool(self.data[model]['regions'])
        # Add Region enabled if valid model
        self.add_btn.config(state=(tk.NORMAL if valid else tk.DISABLED))
        # Delete only enabled if there are regions
        self.del_btn.config(state=(tk.NORMAL if valid and regions_exist else tk.DISABLED))

    def load_model(self):
        self.cancel_draw_mode()
        model = self.model_var.get()
        self.region_listbox.delete(0, tk.END)
        self.canvas.delete("region")
        self.canvas.delete("highlight")
        self.canvas.delete("region_fill")
        self.region_fill_images.clear()
        self._update_model_dependent_buttons()
        if not model or model not in self.data:
            return
        cfg = self.data[model]
        for name in cfg.get("regions", {}):
            self.region_listbox.insert(tk.END, name)
        for name, pts in cfg.get("regions", {}).items():
            coords = [c for p in pts for c in p]
            # draw region outline
            poly_id = self.canvas.create_polygon(*coords, outline="red", width=2, fill="", tags=("region", name))
            # draw semi-transparent fill overlay
            overlay = Image.new("RGBA", (self.frame_w, self.frame_h), (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)
            draw.polygon(coords, fill=(255, 0, 0, 80))
            overlay_tk = ImageTk.PhotoImage(overlay)
            self.canvas.create_image(0, 0, image=overlay_tk, anchor="nw", tags=("region_fill", name))
            self.region_fill_images[name] = overlay_tk

    def setup_ui(self):
        # Fixed-width control panel on left
        ctrl = tk.Frame(self.root, width=140)
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
        self.redo_btn = ttk.Button(ctrl, text="Redo", command=self.redo, state=tk.DISABLED)
        self.redo_btn.pack(fill=tk.X, pady=2)
        self.add_btn = ttk.Button(ctrl, text="Add Region", command=self.enable_draw)
        self.add_btn.pack(fill=tk.X, pady=2)
        self.del_btn = ttk.Button(ctrl, text="Delete Region", command=self.delete_region)
        self.del_btn.pack(fill=tk.X, pady=2)
        # Button to finalize freeform region
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
        # Unbind mouse events and remove preview rectangle
        self.canvas.unbind("<ButtonPress-1>")
        self.canvas.unbind("<B1-Motion>")
        self.canvas.unbind("<ButtonRelease-1>")
        # Remove any in-progress point markers
        for marker in getattr(self, 'point_markers', []):
            self.canvas.delete(marker)
        self.point_markers.clear()
        self.current_points.clear()
        # Bind for region selection
        self.canvas.bind('<Button-1>', self.on_canvas_click)
        if self.rect:
            self.canvas.delete(self.rect)
            self.rect = None
        # disable finish button when exiting draw mode
        self.finish_btn.config(state=tk.DISABLED)

    def rename_model(self):
        self.push_undo()
        model = self.model_var.get()
        if not model or model not in self.data:
            messagebox.showwarning("Warning", "Select a valid model first.")
            return
        new_name = simpledialog.askstring("Rename Model", f"Enter new name for model '{model}':", initialvalue=model)
        if not new_name or new_name == model:
            return
        if new_name in self.data:
            messagebox.showwarning("Warning", f"Model '{new_name}' already exists.")
            return
        # Rename model key
        self.data[new_name] = self.data.pop(model)
        self.save_json()
        models = list(self.data.keys())
        self.model_menu['values'] = models
        self.model_var.set(new_name)
        self.load_model()

    def delete_model(self):
        self.push_undo()
        self.cancel_draw_mode()
        model = self.model_var.get()
        if not model or model not in self.data:
            messagebox.showwarning("Warning", "Select a valid model first.")
            return
        del self.data[model]
        self.save_json()
        models = list(self.data.keys())
        self.model_menu['values'] = models
        if models:
            self.model_var.set(models[0])
        else:
            self.model_var.set('')
        self.load_model()

    def on_region_select(self, event):
        # Highlight selected region
        selection = self.region_listbox.curselection()
        if not selection:
            self.region_info.config(state='normal')
            self.region_info.delete('1.0', tk.END)
            self.region_info.config(state='disabled')
            return
        region_name = self.region_listbox.get(selection[0])
        # clear highlights
        self.canvas.delete('highlight')
        self.canvas.delete('highlight_fill')
        pts = self.data[self.model_var.get()]['regions'].get(region_name)
        if pts:
            coords = [c for p in pts for c in p]
            # draw highlight outline
            self.canvas.create_polygon(*coords, outline='green', width=3, fill="", tags='highlight')
            # draw semi-transparent highlight overlay
            overlay_h = Image.new("RGBA", (self.frame_w, self.frame_h), (0, 0, 0, 0))
            draw_h = ImageDraw.Draw(overlay_h)
            draw_h.polygon(coords, fill=(0, 255, 0, 80))
            overlay_h_tk = ImageTk.PhotoImage(overlay_h)
            self.canvas.create_image(0, 0, image=overlay_h_tk, anchor='nw', tags=("highlight_fill",))
            self.highlight_fill_image = overlay_h_tk
            # Show region info with each coordinate on its own line
            lines = [region_name, 'Points:']
            for x, y in pts:
                lines.append(f"({x}, {y})")
            info = '\n'.join(lines)
            self.region_info.config(state='normal')
            self.region_info.delete('1.0', tk.END)
            self.region_info.insert('1.0', info)
            self.region_info.config(state='disabled')
        else:
            self.region_info.config(state='normal')
            self.region_info.delete('1.0', tk.END)
            self.region_info.config(state='disabled')

    def on_canvas_click(self, event):
        # Use a small box around the click for easier selection
        pad = 4
        items = self.canvas.find_overlapping(event.x - pad, event.y - pad, event.x + pad, event.y + pad)
        # Reverse to prioritize topmost
        items = list(items)[::-1]
        for item in items:
            tags = self.canvas.gettags(item)
            # Look for a region tag that is not just 'region'
            region_tags = [t for t in tags if t != 'region' and t != 'highlight']
            if region_tags:
                region_name = region_tags[0]
                # Select in listbox
                for idx in range(self.region_listbox.size()):
                    if self.region_listbox.get(idx) == region_name:
                        self.region_listbox.select_clear(0, tk.END)
                        self.region_listbox.select_set(idx)
                        self.region_listbox.event_generate('<<ListboxSelect>>')
                        return

    def delete_region(self):
        self.push_undo()
        model = self.model_var.get()
        if not model or model not in self.data:
            messagebox.showwarning("Warning", "Select a valid model first.")
            return
        selection = self.region_listbox.curselection()
        if not selection:
            messagebox.showwarning("Warning", "Select a region first.")
            return
        region = self.region_listbox.get(selection[0])
        del self.data[model]['regions'][region]
        
        # Always clear highlights before reloading the model
        self.canvas.delete('highlight')
        self.canvas.delete('highlight_fill')
        
        # Clear the region info text
        self.region_info.config(state='normal')
        self.region_info.delete('1.0', tk.END)
        self.region_info.config(state='disabled')
        
        self.save_json()
        self.load_model()



    def enable_draw(self):
        self.push_undo()
        model = self.model_var.get()
        if not model:
            messagebox.showwarning("Warning", "Select or create a model first.")
            return
        # Unbind region selection click
        self.canvas.unbind('<Button-1>')
        # Prepare for quadrilateral point selection mode
        for marker in self.point_markers:
            self.canvas.delete(marker)
        self.point_markers = []
        self.current_points = []
        # Bind click for point collection
        self.canvas.bind("<Button-1>", self.on_point_click)
        # enable finish button in draw mode
        self.finish_btn.config(state=tk.NORMAL)

    def on_point_click(self, event):
        x, y = event.x, event.y
        # record point
        self.current_points.append((x, y))
        # draw marker
        r = 5
        marker = self.canvas.create_oval(x-r, y-r, x+r, y+r, fill="blue", tags="point_marker")
        self.point_markers.append(marker)

    def finish_polygon(self):
        # require at least 3 points to finish region
        if len(self.current_points) < 3:
            messagebox.showwarning("Warning", "Need at least 3 points to finish a region.")
            return
        model = self.model_var.get()
        if not model or model not in self.data:
            return
        # clamp and sort points to form non-self-intersecting polygon
        pts = [(max(0, min(x, self.frame_w-1)), max(0, min(y, self.frame_h-1))) for x, y in self.current_points]
        import math
        cx = sum(x for x, y in pts) / len(pts)
        cy = sum(y for x, y in pts) / len(pts)
        sorted_pts = sorted(pts, key=lambda p: math.atan2(p[1] - cy, p[0] - cx))
        self.current_points = sorted_pts
        # save region
        self.push_undo()
        existing = self.data[model]['regions']
        next_num = 1
        while f"region-{next_num}" in existing:
            next_num += 1
        region_name = f"region-{next_num}"
        self.data[model]['regions'] = {region_name: self.current_points.copy()}
        self.save_json()
        # cleanup point markers
        for m in self.point_markers:
            self.canvas.delete(m)
        self.point_markers = []
        self.current_points = []
        # exit draw mode and refresh UI
        self.cancel_draw_mode()
        self.load_model()
        # finalize button state and undo/redo
        self.finish_btn.config(state=tk.DISABLED)
        self._update_undo_redo_buttons()

    def _update_undo_redo_buttons(self):
        if hasattr(self, 'undo_btn'):
            self.undo_btn.config(state=(tk.NORMAL if self.undo_stack else tk.DISABLED))
        if hasattr(self, 'redo_btn'):
            self.redo_btn.config(state=(tk.NORMAL if self.redo_stack else tk.DISABLED))



    def _on_ctrl_z(self, event=None):
        self.undo()
        return "break"

    def _on_ctrl_y(self, event=None):
        self.redo()
        return "break"

    def _on_n(self, event=None):
        if self.add_btn.instate(['!disabled']):
            self.enable_draw()
        return "break"

    def _on_delete(self, event=None):
        if self.del_btn.instate(['!disabled']) and self.region_listbox.curselection():
            self.delete_region()
        return "break"

    def undo(self):
        if not self.undo_stack:
            return
        import copy
        self.redo_stack.append((copy.deepcopy(self.data), self.model_var.get()))
        self.data, model = self.undo_stack.pop()
        self.model_var.set(model)
        self.save_json()
        self.load_model()
        self._update_undo_redo_buttons()

    def redo(self):
        if not self.redo_stack:
            return
        import copy
        self.undo_stack.append((copy.deepcopy(self.data), self.model_var.get()))
        self.data, model = self.redo_stack.pop()
        self.model_var.set(model)
        self.save_json()
        self.load_model()
        self._update_undo_redo_buttons()

    def on_motion(self, event):
        # Clamp both start and end positions to canvas bounds
        MARGIN = 4
        x1 = min(max(self.start_x, MARGIN), self.frame_w - 1 - MARGIN)
        y1 = min(max(self.start_y, MARGIN), self.frame_h - 1 - MARGIN)
        x2 = min(max(event.x, MARGIN), self.frame_w - 1)
        y2 = min(max(event.y, MARGIN), self.frame_h - 1)
        left = min(x1, x2)
        right = max(x1, x2)
        top = min(y1, y2)
        bottom = max(y1, y2)
        self.canvas.coords(self.rect, left, top, right, bottom)

    def on_release(self, event):
        # Check that model is still valid before adding region
        model = self.model_var.get()
        if not model or model not in self.data:
            self.cancel_draw_mode()
            return
        self.push_undo()
        # Finish region
        x1, y1 = self.start_x, self.start_y
        x2 = min(max(event.x, 0), self.frame_w - 1)
        y2 = min(max(event.y, 0), self.frame_h - 1)
        left = min(x1, x2)
        right = max(x1, x2)
        top = min(y1, y2)
        bottom = max(y1, y2)
        rect = [(left, top), (right, top), (right, bottom), (left, bottom)]

        # Generate unique region name
        existing = self.data[model]['regions']
        next_num = 1
        while f"region-{next_num}" in existing:
            next_num += 1
        region_name = f"region-{next_num}"
        self.data[model]['regions'][region_name] = rect
        self.save_json()
        self.canvas.delete("drawing")
        # Unbind drawing events
        self.canvas.unbind("<ButtonPress-1>")
        self.canvas.unbind("<B1-Motion>")
        self.canvas.unbind("<ButtonRelease-1>")
        # Bind for region selection
        self.canvas.bind('<Button-1>', self.on_canvas_click)
        self.load_model()
        self._update_undo_redo_buttons()

    def update_webcam(self):
        ret, frame = self.cap.read()
        if ret:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.bg_image = ImageTk.PhotoImage(Image.fromarray(img))
            self.canvas.create_image(0, 0, image=self.bg_image, anchor='nw')
        self.root.after(33, self.update_webcam)  # ~30 FPS

    def on_close(self):
        self.save_json()
        messagebox.showinfo("Saved", f"Region saved to {REGIONS_JSON_PATH}.")
        self.cap.release()
        self.root.destroy()

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