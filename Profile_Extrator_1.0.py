"""
Graph Data Extractor Pro
"""

# ================================
# Part 1: Main Application and UI Setup
# ================================

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.colorchooser import askcolor
from PIL import Image, ImageTk, ImageDraw
import pandas as pd
import threading
import queue
import logging
import os
import pickle
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class GraphDataExtractorPro:
    def __init__(self, master):
        self.master = master
        self.master.title("Graph Data Extractor Pro")
        self.master.geometry("1200x900")
        self.master.configure(bg='#f0f0f0')

        # Configure logging
        logging.basicConfig(level=logging.INFO, filename='app.log',
                            format='%(asctime)s - %(levelname)s - %(message)s')

        # Initialize variables
        self.init_variables()

        # Create manager instances BEFORE creating UI components
        self.image_manager = ImageManager(self)
        self.data_manager = DataManager(self)

        # Create UI components
        self.create_menu()
        self.create_control_panel()
        self.create_canvas()
        self.create_status_bar()

        # Bind events
        self.bind_events()

        # Queue for thread-safe GUI updates
        self.queue = queue.Queue()
        self.master.after(100, self.process_queue)

    def init_variables(self):
        self.status_var = tk.StringVar()
        self.undo_stack = []
        self.redo_stack = []
        self.point_color = 'red'
        self.axis_colors = {'x': 'blue', 'y': 'green'}
        self.x_limits = None
        self.y_limits = None

    def create_menu(self):
        menubar = tk.Menu(self.master)
        self.master.config(menu=menubar)

        # File Menu
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Load Image", accelerator="Ctrl+O", command=self.load_image)
        file_menu.add_command(label="Save Image", accelerator="Ctrl+S", command=self.save_image)
        file_menu.add_command(label="New Image", accelerator="Ctrl+N", command=self.new_image)
        file_menu.add_separator()
        file_menu.add_command(label="Save Session", command=self.save_session)
        file_menu.add_command(label="Load Session", command=self.load_session)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.master.quit)
        menubar.add_cascade(label="File", menu=file_menu)

        # Edit Menu
        edit_menu = tk.Menu(menubar, tearoff=0)
        edit_menu.add_command(label="Undo", accelerator="Ctrl+Z", command=self.undo)
        edit_menu.add_command(label="Redo", accelerator="Ctrl+Y", command=self.redo)
        edit_menu.add_separator()
        edit_menu.add_command(label="Clear All", command=self.clear_canvas)
        menubar.add_cascade(label="Edit", menu=edit_menu)

        # View Menu
        view_menu = tk.Menu(menubar, tearoff=0)
        view_menu.add_command(label="Zoom In", accelerator="Ctrl++", command=self.zoom_in)
        view_menu.add_command(label="Zoom Out", accelerator="Ctrl+-", command=self.zoom_out)
        view_menu.add_command(label="Reset Zoom", command=self.reset_zoom)
        menubar.add_cascade(label="View", menu=view_menu)

        # Help Menu
        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="Documentation", command=self.show_documentation)
        help_menu.add_command(label="About", command=self.show_about)
        menubar.add_cascade(label="Help", menu=help_menu)

        # Keyboard Shortcuts
        self.master.bind_all("<Control-o>", lambda event: self.load_image())
        self.master.bind_all("<Control-s>", lambda event: self.save_image())
        self.master.bind_all("<Control-n>", lambda event: self.new_image())
        self.master.bind_all("<Control-z>", lambda event: self.undo())
        self.master.bind_all("<Control-y>", lambda event: self.redo())
        self.master.bind_all("<Control-plus>", lambda event: self.zoom_in())
        self.master.bind_all("<Control-minus>", lambda event: self.zoom_out())

    def create_control_panel(self):
        control_frame = ttk.Frame(self.master)
        control_frame.pack(fill=tk.X, padx=10, pady=5)

        notebook = ttk.Notebook(control_frame)
        notebook.pack(fill=tk.X)

        # File Operations Tab
        file_tab = ttk.Frame(notebook)
        notebook.add(file_tab, text="File Operations")

        ttk.Button(file_tab, text="📂 Load Image", command=self.load_image,
                   style='Custom.TButton').pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Button(file_tab, text="💾 Save Image", command=self.save_image,
                   style='Custom.TButton').pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Button(file_tab, text="🆕 New Image", command=self.new_image,
                   style='Custom.TButton').pack(side=tk.LEFT, padx=5, pady=5)

        # Axis Setup Tab
        axis_tab = ttk.Frame(notebook)
        notebook.add(axis_tab, text="Axis Setup")

        ttk.Button(axis_tab, text="↔️ Set X-Axis", command=self.start_set_x_axis,
                   style='Custom.TButton').pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Button(axis_tab, text="↕️ Set Y-Axis", command=self.start_set_y_axis,
                   style='Custom.TButton').pack(side=tk.LEFT, padx=5, pady=5)

        # Data Operations Tab
        data_tab = ttk.Frame(notebook)
        notebook.add(data_tab, text="Data Operations")

        ttk.Button(data_tab, text="📊 Extract Data", command=self.extract_data,
                   style='Custom.TButton').pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Button(data_tab, text="📈 Plot Data", command=self.plot_data,
                   style='Custom.TButton').pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Button(data_tab, text="✏️ Edit/Delete Point", command=self.edit_delete_point,
                   style='Custom.TButton').pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Button(data_tab, text="🗑️ Clear All", command=self.clear_canvas,
                   style='Custom.TButton').pack(side=tk.LEFT, padx=5, pady=5)

        # Color Customization Tab
        color_tab = ttk.Frame(notebook)
        notebook.add(color_tab, text="Customization")

        ttk.Button(color_tab, text="Choose Point Color", command=self.choose_point_color,
                   style='Custom.TButton').pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Button(color_tab, text="Choose X-Axis Color", command=lambda: self.choose_axis_color("x"),
                   style='Custom.TButton').pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Button(color_tab, text="Choose Y-Axis Color", command=lambda: self.choose_axis_color("y"),
                   style='Custom.TButton').pack(side=tk.LEFT, padx=5, pady=5)

        # Create input frame for axis limits
        input_frame = ttk.Frame(control_frame)
        input_frame.pack(fill=tk.X, pady=5)

        # Axis Limits Frame
        limits_frame = ttk.Frame(input_frame)
        limits_frame.pack(fill=tk.X, pady=5)

        # X-Axis limits
        x_limits_group = ttk.LabelFrame(limits_frame, text="X-Axis Limits", padding=5)
        x_limits_group.pack(side=tk.LEFT, padx=5)

        ttk.Label(x_limits_group, text="Initial:").pack(side=tk.LEFT)
        self.x_init_entry = ttk.Entry(x_limits_group, width=10)
        self.x_init_entry.pack(side=tk.LEFT, padx=5)

        ttk.Label(x_limits_group, text="Final:").pack(side=tk.LEFT)
        self.x_final_entry = ttk.Entry(x_limits_group, width=10)
        self.x_final_entry.pack(side=tk.LEFT, padx=5)

        # Y-Axis limits
        y_limits_group = ttk.LabelFrame(limits_frame, text="Y-Axis Limits", padding=5)
        y_limits_group.pack(side=tk.LEFT, padx=5)

        ttk.Label(y_limits_group, text="Initial:").pack(side=tk.LEFT)
        self.y_init_entry = ttk.Entry(y_limits_group, width=10)
        self.y_init_entry.pack(side=tk.LEFT, padx=5)

        ttk.Label(y_limits_group, text="Final:").pack(side=tk.LEFT)
        self.y_final_entry = ttk.Entry(y_limits_group, width=10)
        self.y_final_entry.pack(side=tk.LEFT, padx=5)

        ttk.Button(input_frame, text="✓ Apply Limits", command=self.set_limits,
                   style='Custom.TButton').pack(side=tk.LEFT, padx=5)

    # Delegate Methods
    def load_image(self):
        """Delegate to ImageManager"""
        self.image_manager.load_image()

    def save_image(self):
        """Delegate to ImageManager"""
        self.image_manager.save_image()

    def new_image(self):
        """Clear current image and load a new one"""
        confirm = messagebox.askyesno("New Image", "Are you sure you want to load a new image?\nUnsaved data will be lost.")
        if confirm:
            self.clear_canvas()
            self.image_manager.load_image()

    def zoom_in(self):
        """Delegate to ImageManager"""
        self.image_manager.zoom_in()

    def zoom_out(self):
        """Delegate to ImageManager"""
        self.image_manager.zoom_out()

    def reset_zoom(self):
        """Delegate to ImageManager"""
        self.image_manager.reset_zoom()

    def clear_canvas(self):
        """Delegate to DataManager"""
        self.data_manager.clear_canvas()

    def extract_data(self):
        """Delegate to DataManager"""
        self.data_manager.extract_data()

    def plot_data(self):
        """Delegate to DataManager"""
        self.data_manager.plot_data()

    def start_set_x_axis(self):
        """Delegate to DataManager"""
        self.data_manager.start_set_x_axis()

    def start_set_y_axis(self):
        """Delegate to DataManager"""
        self.data_manager.start_set_y_axis()

    def edit_delete_point(self):
        """Delegate to DataManager"""
        self.data_manager.edit_delete_point()

    def choose_point_color(self):
        """Delegate to DataManager"""
        self.data_manager.choose_point_color()

    def choose_axis_color(self, axis):
        """Delegate to DataManager"""
        self.data_manager.choose_axis_color(axis)

    def set_limits(self):
        """Delegate to DataManager"""
        self.data_manager.set_limits()

    def save_session(self):
        """Delegate to DataManager"""
        self.data_manager.save_session()

    def load_session(self):
        """Delegate to DataManager"""
        self.data_manager.load_session()

    def undo(self):
        """Delegate to DataManager"""
        self.data_manager.undo()

    def redo(self):
        """Delegate to DataManager"""
        self.data_manager.redo()

    def create_canvas(self):
        # Create main container with all controls at top
        self.main_container = ttk.Frame(self.master)
        self.main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Define style
        self.style = ttk.Style()
        self.style.configure('Custom.TButton', padding=5, font=('Helvetica', 10))
        self.style.configure('Custom.TFrame', background='#f0f0f0')
        self.style.configure('Canvas.TFrame', relief='solid', borderwidth=1)

        # Create canvas frame
        self.image_manager.create_canvas_frame()

    def create_status_bar(self):
        status_frame = ttk.Frame(self.master)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X)
        status_label = ttk.Label(status_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_label.pack(fill=tk.X)
        self.status_var.set("Ready")

    def bind_events(self):
        self.master.bind("<Configure>", self.on_resize)

    def process_queue(self):
        try:
            while True:
                task, data = self.queue.get_nowait()
                if task == "load_image":
                    self.image_manager.image = data
                    self.image_manager.display_image()
                    self.data_manager.redraw_all()
                    self.status_var.set("Image loaded successfully.")
                elif task == "error":
                    messagebox.showerror("Error", data)
                    self.status_var.set("Error occurred.")
        except queue.Empty:
            pass
        self.master.after(100, self.process_queue)

    def show_about(self):
        messagebox.showinfo("About", "Graph Data Extractor Pro\nVersion 2.0\nDeveloped by Your Name")

    def show_documentation(self):
        messagebox.showinfo("Documentation", "User Guide:\n1. Load an image.\n2. Set the axes.\n3. Click to add data points.\n4. Extract or plot data.")

    def on_resize(self, event):
        # Handle window resize if necessary
        pass
# ================================
# Part 2: Image and Canvas Management
# ================================

class ImageManager:
    def __init__(self, app):
        self.app = app
        self.canvas = None
        self.canvas_frame = None
        self.image = None
        self.tk_image = None
        self.zoom_factor = 1.0
        self.image_id = None
        self.canvas_items = {'points': [], 'axis_lines': []}
        self.selected_point = None
        self.image_offset = (0, 0)

    def create_canvas_frame(self):
        self.canvas_frame = ttk.Frame(self.app.main_container, style='Canvas.TFrame')
        self.canvas_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Create scrollbars
        self.v_scroll = ttk.Scrollbar(self.canvas_frame, orient=tk.VERTICAL)
        self.v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.h_scroll = ttk.Scrollbar(self.canvas_frame, orient=tk.HORIZONTAL)
        self.h_scroll.pack(side=tk.BOTTOM, fill=tk.X)

        # Create canvas
        self.canvas = tk.Canvas(self.canvas_frame, bg='white',
                                highlightthickness=0,
                                xscrollcommand=self.h_scroll.set,
                                yscrollcommand=self.v_scroll.set)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.v_scroll.config(command=self.canvas.yview)
        self.h_scroll.config(command=self.canvas.xview)

        # Bind canvas events
        self.canvas.bind("<Button-1>", self.on_left_click)
        self.canvas.bind("<Button-3>", self.on_right_click)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

    def load_image_thread(self, file_path):
        try:
            image = Image.open(file_path).convert("RGB")
            self.app.queue.put(("load_image", image))
            logging.info(f"Image loaded successfully from {file_path}")
        except Exception as e:
            self.app.queue.put(("error", f"Failed to load image: {e}"))
            logging.error(f"Failed to load image: {e}")

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.png;*.bmp;*.gif")])
        if file_path:
            threading.Thread(target=self.load_image_thread, args=(file_path,), daemon=True).start()

    def display_image(self):
        if self.image:
            # Get canvas dimensions
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()

            # Resize image based on zoom factor
            width = int(self.image.width * self.zoom_factor)
            height = int(self.image.height * self.zoom_factor)
            resized_image = self.image.resize((width, height), Image.LANCZOS)
            self.tk_image = ImageTk.PhotoImage(resized_image)

            # Calculate center position
            x_center = (canvas_width - width) // 2
            y_center = (canvas_height - height) // 2

            # Clear previous image
            if self.image_id:
                self.canvas.delete(self.image_id)

            # Add image to canvas at center
            self.image_id = self.canvas.create_image(x_center, y_center, anchor=tk.NW, image=self.tk_image, tags="image")
            self.canvas.config(scrollregion=self.canvas.bbox(tk.ALL))

            # Store the offset for point calculations
            self.image_offset = (x_center, y_center)

    def redraw_all(self):
        # Clear existing points and lines
        self.canvas.delete("point")
        self.canvas.delete("line")

        # Redraw points
        for point in self.app.data_manager.points:
            x, y = self.scale_coordinates(point)
            item = self.canvas.create_oval(x - 3, y - 3, x + 3, y + 3,
                                           fill=self.app.point_color, tags="point")
            self.canvas_items['points'].append(item)

        # Redraw axis lines
        for axis in self.app.data_manager.axis_lines:
            x1, y1 = self.scale_coordinates((axis[0], axis[1]))
            x2, y2 = self.scale_coordinates((axis[2], axis[3]))
            color = self.app.axis_colors.get(axis[4], 'black')
            item = self.canvas.create_line(x1, y1, x2, y2,
                                           fill=color, width=2, tags="line")
            self.canvas_items['axis_lines'].append(item)

    def scale_coordinates(self, point):
        if hasattr(self, 'image_offset'):
            x, y = point
            scaled_x = (x * self.zoom_factor) + self.image_offset[0]
            scaled_y = (y * self.zoom_factor) + self.image_offset[1]
            return scaled_x, scaled_y
        return point[0] * self.zoom_factor, point[1] * self.zoom_factor

    def on_left_click(self, event):
        if not self.image:
            return  # No image loaded

        # Get canvas coordinates
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)

        # Adjust coordinates relative to image
        if hasattr(self, 'image_offset'):
            x = (x - self.image_offset[0]) / self.zoom_factor
            y = (y - self.image_offset[1]) / self.zoom_factor
        else:
            return  # No image loaded

        # If we're drawing an axis, add the point
        if self.app.data_manager.drawing_axis:
            self.app.data_manager.add_axis_point((x, y))
            return

        # Otherwise add a regular data point
        self.app.data_manager.add_point((x, y))
        self.app.undo_stack.append(('add', (x, y)))
        self.app.redo_stack.clear()
        self.app.status_var.set("Point added")
        logging.info(f"Point added at ({x:.2f}, {y:.2f})")

    def on_right_click(self, event):
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)

        # Adjust coordinates relative to image
        if hasattr(self, 'image_offset'):
            x = (x - self.image_offset[0]) / self.zoom_factor
            y = (y - self.image_offset[1]) / self.zoom_factor
        else:
            return  # No image loaded

        self.app.data_manager.select_point(x, y)
        if self.app.data_manager.selected_point:
            self.highlight_selected_point()
        else:
            self.app.status_var.set("No point selected.")

    def on_drag(self, event):
        if self.app.data_manager.selected_point:
            x = self.canvas.canvasx(event.x)
            y = self.canvas.canvasy(event.y)
            self.app.data_manager.move_point(x, y)
            self.redraw_all()

    def on_release(self, event):
        if self.app.data_manager.selected_point:
            self.app.undo_stack.append(('move', self.app.data_manager.previous_position, self.app.data_manager.selected_point))
            self.app.redo_stack.clear()
            self.app.data_manager.previous_position = None
            self.app.status_var.set("Point moved.")
            logging.info("Point moved.")

    def highlight_selected_point(self):
        self.canvas.delete("highlight")
        point = self.app.data_manager.selected_point
        if point:
            x, y = self.scale_coordinates(point)
            self.canvas.create_oval(x - 5, y - 5, x + 5, y + 5, outline='yellow', width=2, tags="highlight")
            self.app.status_var.set(f"Selected Point at ({x:.2f}, {y:.2f})")
            logging.info(f"Highlighted selected point at ({x:.2f}, {y:.2f})")

    def zoom_in(self):
        self.zoom_factor *= 1.1
        self.display_image()
        self.redraw_all()
        self.app.status_var.set(f"Zoomed In to {self.zoom_factor:.2f}x")
        logging.info(f"Zoomed In to {self.zoom_factor:.2f}x")

    def zoom_out(self):
        self.zoom_factor /= 1.1
        self.display_image()
        self.redraw_all()
        self.app.status_var.set(f"Zoomed Out to {self.zoom_factor:.2f}x")
        logging.info(f"Zoomed Out to {self.zoom_factor:.2f}x")

    def reset_zoom(self):
        self.zoom_factor = 1.0
        self.display_image()
        self.redraw_all()
        self.app.status_var.set("Zoom reset to 1.0x")
        logging.info("Zoom reset to 1.0x")

    def choose_image_color(self, color):
        self.app.point_color = color
        self.redraw_all()
        self.app.status_var.set(f"Point color changed.")
        logging.info(f"Point color changed to {color}.")

    def save_image(self):
        if self.image:
            file_path = filedialog.asksaveasfilename(defaultextension=".png",
                                                     filetypes=[("PNG files", "*.png"),
                                                                ("JPEG files", "*.jpg;*.jpeg"),
                                                                ("All files", "*.*")])
            if file_path:
                try:
                    # Create a temporary image with all drawings
                    temp_image = self.image.copy()
                    draw = ImageDraw.Draw(temp_image)

                    # Draw points
                    for point in self.app.data_manager.points:
                        x, y = self.map_graph_to_pixel(point, self.app.x_limits, self.app.y_limits)
                        draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=self.app.point_color)

                    # Draw axis lines
                    for axis in self.app.data_manager.axis_lines:
                        x1, y1 = self.map_graph_to_pixel((axis[0], axis[1]), self.app.x_limits, self.app.y_limits)
                        x2, y2 = self.map_graph_to_pixel((axis[2], axis[3]), self.app.x_limits, self.app.y_limits)
                        color = self.app.axis_colors.get(axis[4], 'black')
                        draw.line((x1, y1, x2, y2), fill=color, width=2)

                    temp_image.save(file_path)
                    messagebox.showinfo("Image Saved", f"Image saved to {file_path}")
                    logging.info(f"Image saved successfully to {file_path}")
                except Exception as e:
                    messagebox.showerror("Save Error", f"Failed to save image: {e}")
                    logging.error(f"Image save failed: {e}")
        else:
            messagebox.showwarning("No Image", "Please load an image first.")
            logging.warning("Save attempted without an image loaded.")

    def map_graph_to_pixel(self, point, x_limits, y_limits):
        """
        Maps graph coordinates to pixel coordinates based on axis limits.
        """
        x_graph, y_graph = point
        if not x_limits or not y_limits:
            return 0, 0
        x_pixel = ((x_graph - x_limits[0]) / (x_limits[1] - x_limits[0])) * self.image.width
        y_pixel = self.image.height - ((y_graph - y_limits[0]) / (y_limits[1] - y_limits[0])) * self.image.height  # Inverted Y-axis
        return x_pixel, y_pixel

    def on_resize(self, event):
        # Handle window resize if necessary
        pass
# ================================
# Part 3: Data Handling, Export, and Utility Functions
# ================================

class DataManager:
    def __init__(self, app):
        self.app = app
        self.points = []  # List of tuples (x, y)
        self.axis_lines = []  # List of tuples (x1, y1, x2, y2, axis_type)
        self.drawing_axis = None  # 'x' or 'y' when setting axes
        self.axis_points = []  # Temporary storage for axis points
        self.selected_point = None  # Currently selected point
        self.previous_position = None  # Previous position of the selected point for undo

    def add_axis_point(self, point):
        """Add a point when setting axis lines."""
        if not self.drawing_axis:
            return

        # Store the original point coordinates
        self.axis_points.append(point)

        # Calculate scaled coordinates for drawing
        scaled_x, scaled_y = self.app.image_manager.scale_coordinates(point)

        # Create a visible dot where clicked
        color = self.app.axis_colors.get(self.drawing_axis, 'black')
        dot_id = self.app.image_manager.canvas.create_oval(
            scaled_x - 4, scaled_y - 4,
            scaled_x + 4, scaled_y + 4,
            fill=color,
            outline=color,
            tags="axis_point"
        )

        if len(self.axis_points) == 2:
            x1, y1 = self.axis_points[0]
            x2, y2 = self.axis_points[1]

            # Add the axis line with the original coordinates
            self.axis_lines.append((x1, y1, x2, y2, self.drawing_axis))

            # Clear temporary dots
            self.app.image_manager.canvas.delete("axis_point")

            # Redraw everything
            self.app.image_manager.redraw_all()

            # Reset axis drawing state
            self.reset_axis_drawing()

            # Log the action
            logging.info(f"Added {self.drawing_axis}-axis line from ({x1:.2f}, {y1:.2f}) to ({x2:.2f}, {y2:.2f})")
            self.app.status_var.set(f"{self.drawing_axis.upper()}-axis set.")

    def add_point(self, point):
        """Add a data point."""
        self.points.append(point)
        self.app.image_manager.redraw_all()
        logging.info(f"Added point at {point}")

    def select_point(self, x, y):
        """Select a point near the (x, y) coordinates."""
        tolerance = 10 / self.app.image_manager.zoom_factor
        for point in self.points:
            px, py = point
            if abs(px - x) < tolerance and abs(py - y) < tolerance:
                self.selected_point = point
                self.previous_position = point
                logging.info(f"Point selected: {point}")
                return
        self.selected_point = None
        logging.info("No point selected.")

    def move_point(self, new_x, new_y):
        """Move the selected point to new coordinates."""
        if self.selected_point:
            if hasattr(self.app.image_manager, 'image_offset'):
                adjusted_x = (new_x - self.app.image_manager.image_offset[0]) / self.app.image_manager.zoom_factor
                adjusted_y = (new_y - self.app.image_manager.image_offset[1]) / self.app.image_manager.zoom_factor
            else:
                adjusted_x = new_x / self.app.image_manager.zoom_factor
                adjusted_y = new_y / self.app.image_manager.zoom_factor

            index = self.points.index(self.selected_point)
            self.points[index] = (adjusted_x, adjusted_y)
            self.selected_point = self.points[index]

            logging.info(f"Moved point to: ({adjusted_x:.2f}, {adjusted_y:.2f})")
            self.app.image_manager.redraw_all()

    def start_set_x_axis(self):
        """Initiate the process to set the X-axis."""
        self.drawing_axis = 'x'
        self.axis_points = []
        # Clear any existing temporary points
        self.app.image_manager.canvas.delete("axis_point")
        self.app.status_var.set("Click two points to set X-axis: Point 1/2")
        messagebox.showinfo("Set X-Axis", "Click two points on the graph to set the X-axis.\nA dot will appear for each click.")

    def start_set_y_axis(self):
        """Initiate the process to set the Y-axis."""
        self.drawing_axis = 'y'
        self.axis_points = []
        # Clear any existing temporary points
        self.app.image_manager.canvas.delete("axis_point")
        self.app.status_var.set("Click two points to set Y-axis: Point 1/2")
        messagebox.showinfo("Set Y-Axis", "Click two points on the graph to set the Y-axis.\nA dot will appear for each click.")

    def reset_axis_drawing(self):
        """Reset axis drawing state."""
        self.drawing_axis = None
        self.axis_points = []
        # Clear any temporary points
        self.app.image_manager.canvas.delete("axis_point")
        self.app.status_var.set("Axis set successfully")

    def clear_canvas(self):
        """Clear all points and axis lines from the canvas."""
        confirm = messagebox.askyesno("Clear All", "Are you sure you want to clear all points and axis lines?")
        if confirm:
            self.points.clear()
            self.axis_lines.clear()
            self.drawing_axis = None
            self.axis_points = []
            self.selected_point = None
            self.previous_position = None
            self.app.image_manager.redraw_all()
            self.app.undo_stack.clear()
            self.app.redo_stack.clear()
            self.app.status_var.set("Canvas cleared.")
            logging.info("Canvas cleared.")

    def extract_data(self):
        """Extract data points and export them to Excel or CSV."""
        if not self.points:
            messagebox.showwarning("No Points", "No points extracted. Please click on the graph to extract data.")
            return

        if not self.app.x_limits or not self.app.y_limits:
            messagebox.showwarning("No Axis Limits", "Please set axis limits before extracting data.")
            logging.warning("Data extraction attempted without axis limits.")
            return

        # Create dataframe for points
        df = pd.DataFrame(self.points, columns=["X (Pixels)", "Y (Pixels)"])
        x_graph = df["X (Pixels)"].apply(lambda x: self.map_pixel_to_graph(x, self.app.x_limits, self.app.image_manager.image.width))
        y_graph = df["Y (Pixels)"].apply(lambda y: self.map_pixel_to_graph(y, self.app.y_limits, self.app.image_manager.image.height, invert=True))
        df["X (Graph)"] = x_graph
        df["Y (Graph)"] = y_graph

        # Choose export format
        export_format = self.choose_export_format()
        if not export_format:
            return

        # Save file dialog
        file_path = filedialog.asksaveasfilename(defaultextension=f".{export_format}",
                                                 filetypes=[(f"{export_format.upper()} files", f"*.{export_format}")])
        if file_path:
            try:
                if export_format == "xlsx":
                    df.to_excel(file_path, index=False)
                elif export_format == "csv":
                    df.to_csv(file_path, index=False)
                messagebox.showinfo("Data Extracted", f"Data saved to {file_path}")
                logging.info(f"Data exported successfully to {file_path}")
            except Exception as e:
                messagebox.showerror("Export Error", f"Failed to export data: {e}")
                logging.error(f"Data export failed: {e}")

    def choose_export_format(self):
        """Open a dialog to choose the export format (Excel or CSV)."""
        export_window = tk.Toplevel(self.app.master)
        export_window.title("Choose Export Format")
        export_window.geometry("300x150")
        export_window.grab_set()

        selected_format = tk.StringVar(value="xlsx")

        ttk.Label(export_window, text="Select Export Format:").pack(pady=10)
        ttk.Radiobutton(export_window, text="Excel (.xlsx)", variable=selected_format, value="xlsx").pack()
        ttk.Radiobutton(export_window, text="CSV (.csv)", variable=selected_format, value="csv").pack()

        def confirm():
            export_window.destroy()

        ttk.Button(export_window, text="Confirm", command=confirm).pack(pady=10)
        self.app.master.wait_window(export_window)
        return selected_format.get()

    def map_pixel_to_graph(self, pixel, limits, max_pixel, invert=False):
        """
        Maps pixel coordinates to graph coordinates based on axis limits.

        Args:
            pixel (float): The pixel value to map.
            limits (tuple): The (initial, final) limits of the axis.
            max_pixel (int): The maximum pixel value corresponding to the final limit.
            invert (bool): Whether to invert the axis (useful for Y-axis).

        Returns:
            float: The corresponding graph coordinate.
        """
        initial, final = limits
        value = initial + (pixel / max_pixel) * (final - initial)
        if invert:
            value = final - (pixel / max_pixel) * (final - initial)
        return value

    def undo(self):
        """Undo the last action."""
        if not self.app.undo_stack:
            messagebox.showinfo("Undo", "No actions to undo.")
            logging.info("Undo attempted with empty undo stack.")
            return
        action = self.app.undo_stack.pop()
        self.app.redo_stack.append(action)
        action_type = action[0]

        if action_type == 'add':
            point = action[1]
            if point in self.points:
                self.points.remove(point)
            self.app.image_manager.redraw_all()
            self.app.status_var.set("Undo: Removed last point.")
            logging.info("Undo: Removed last point.")

        elif action_type == 'move':
            previous_pos, point = action[1], action[2]
            index = self.points.index(point)
            self.points[index] = previous_pos
            self.app.image_manager.redraw_all()
            self.app.status_var.set("Undo: Moved point back to previous position.")
            logging.info("Undo: Moved point back to previous position.")

        elif action_type == 'delete':
            point = action[1]
            self.points.append(point)
            self.app.image_manager.redraw_all()
            self.app.status_var.set("Undo: Re-added deleted point.")
            logging.info("Undo: Re-added deleted point.")

    def redo(self):
        """Redo the last undone action."""
        if not self.app.redo_stack:
            messagebox.showinfo("Redo", "No actions to redo.")
            logging.info("Redo attempted with empty redo stack.")
            return
        action = self.app.redo_stack.pop()
        self.app.undo_stack.append(action)
        action_type = action[0]

        if action_type == 'add':
            point = action[1]
            self.points.append(point)
            self.app.image_manager.redraw_all()
            self.app.status_var.set("Redo: Re-added point.")
            logging.info("Redo: Re-added point.")

        elif action_type == 'move':
            previous_pos, point = action[1], action[2]
            index = self.points.index(point)
            self.points[index] = previous_pos  # This assumes previous_pos is the new position
            self.app.image_manager.redraw_all()
            self.app.status_var.set("Redo: Moved point again.")
            logging.info("Redo: Moved point again.")

        elif action_type == 'delete':
            point = action[1]
            if point in self.points:
                self.points.remove(point)
            self.app.image_manager.redraw_all()
            self.app.status_var.set("Redo: Deleted point again.")
            logging.info("Redo: Deleted point again.")

    def edit_delete_point(self):
        """Delete the currently selected point."""
        if not self.selected_point:
            messagebox.showwarning("No Point Selected", "Right-click on a point to select it for editing or deletion.")
            logging.warning("Edit/Delete attempted without a selected point.")
            return

        confirm = messagebox.askyesno("Delete Point", "Are you sure you want to delete the selected point?")
        if confirm:
            self.points.remove(self.selected_point)
            self.app.image_manager.redraw_all()
            self.app.status_var.set("Point deleted.")
            self.app.undo_stack.append(('delete', self.selected_point))
            self.app.redo_stack.clear()
            logging.info("Edit/Delete: Point deleted.")
            self.selected_point = None

    def choose_point_color(self):
        """Choose and set the color for the data points."""
        color = askcolor(title="Choose color for points")[1]
        if color:
            self.app.point_color = color
            self.app.image_manager.redraw_all()
            self.app.status_var.set(f"Point color changed.")
            logging.info(f"Point color changed to {color}.")

    def choose_axis_color(self, axis):
        """Choose and set the color for the specified axis line."""
        color = askcolor(title=f"Choose color for {axis.upper()}-Axis")[1]
        if color:
            self.app.axis_colors[axis] = color
            self.app.image_manager.redraw_all()
            self.app.status_var.set(f"{axis.upper()}-Axis color changed.")
            logging.info(f"{axis.upper()}-Axis color changed to {color}.")

    def set_limits(self):
        """Set the axis limits based on user input."""
        try:
            x_init = float(self.app.x_init_entry.get())
            x_final = float(self.app.x_final_entry.get())
            y_init = float(self.app.y_init_entry.get())
            y_final = float(self.app.y_final_entry.get())

            if x_init >= x_final:
                raise ValueError("X-Axis: Final value must be greater than initial value.")
            if y_init >= y_final:
                raise ValueError("Y-Axis: Final value must be greater than initial value.")

            self.app.x_limits = (x_init, x_final)
            self.app.y_limits = (y_init, y_final)

            messagebox.showinfo("Limits Set",
                                f"X-Axis Limits: {self.app.x_limits}\n"
                                f"Y-Axis Limits: ({y_init}, {y_final})")
            logging.info(f"Axis limits set: X={self.app.x_limits}, Y={self.app.y_limits}")
        except ValueError as e:
            messagebox.showerror("Input Error", str(e))
            logging.warning(f"Set limits failed: {e}")

    def save_session(self):
        """Save the current session to a file."""
        file_path = filedialog.asksaveasfilename(defaultextension=".gdep",
                                                 filetypes=[("GDEP Session files", "*.gdep")])
        if file_path:
            session_data = {
                'points': self.points,
                'axis_lines': self.axis_lines,
                'x_limits': self.app.x_limits,
                'y_limits': self.app.y_limits,
                'image': self.app.image_manager.image,
                'point_color': self.app.point_color,
                'axis_colors': self.app.axis_colors
            }
            try:
                with open(file_path, 'wb') as f:
                    pickle.dump(session_data, f)
                messagebox.showinfo("Session Saved", f"Session saved to {file_path}")
                logging.info(f"Session saved to {file_path}")
            except Exception as e:
                messagebox.showerror("Save Error", f"Failed to save session: {e}")
                logging.error(f"Session save failed: {e}")

    def load_session(self):
        """Load a session from a file."""
        file_path = filedialog.askopenfilename(filetypes=[("GDEP Session files", "*.gdep")])
        if file_path:
            try:
                with open(file_path, 'rb') as f:
                    session_data = pickle.load(f)
                self.points = session_data.get('points', [])
                self.axis_lines = session_data.get('axis_lines', [])
                self.app.x_limits = session_data.get('x_limits')
                self.app.y_limits = session_data.get('y_limits')
                self.app.image_manager.image = session_data.get('image')
                self.app.point_color = session_data.get('point_color', 'red')
                self.app.axis_colors = session_data.get('axis_colors', {'x': 'blue', 'y': 'green'})
                self.app.image_manager.display_image()
                self.redraw_all()
                messagebox.showinfo("Session Loaded", f"Session loaded from {file_path}")
                logging.info(f"Session loaded from {file_path}")
            except Exception as e:
                messagebox.showerror("Load Error", f"Failed to load session: {e}")
                logging.error(f"Session load failed: {e}")

    def redraw_all(self):
        """Delegate to ImageManager"""
        self.app.image_manager.redraw_all()

    def plot_data(self):
        """Plot the extracted data within the application."""
        if not self.points:
            messagebox.showwarning("No Data", "No data to plot. Please extract data points first.")
            return

        if not self.app.x_limits or not self.app.y_limits:
            messagebox.showwarning("No Axis Limits", "Please set axis limits before plotting data.")
            return

        # Create dataframe for points
        df = pd.DataFrame(self.points, columns=["X (Pixels)", "Y (Pixels)"])
        x_graph = df["X (Pixels)"].apply(lambda x: self.map_pixel_to_graph(x, self.app.x_limits, self.app.image_manager.image.width))
        y_graph = df["Y (Pixels)"].apply(lambda y: self.map_pixel_to_graph(y, self.app.y_limits, self.app.image_manager.image.height, invert=True))
        df["X (Graph)"] = x_graph
        df["Y (Graph)"] = y_graph

        # Plot data
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(df["X (Graph)"], df["Y (Graph)"], marker='o', linestyle='-', color=self.app.point_color)
        ax.set_xlabel('X-Axis')
        ax.set_ylabel('Y-Axis')
        ax.set_title('Extracted Data Plot')
        ax.grid(True)

        # Create a new window for the plot
        plot_window = tk.Toplevel(self.app.master)
        plot_window.title("Data Plot")
        plot_window.geometry("700x500")

        canvas = FigureCanvasTkAgg(fig, master=plot_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        logging.info("Data plotted successfully.")
# ================================
# Part 4: Application Entry Point
# ================================

if __name__ == "__main__":
    root = tk.Tk()
    app = GraphDataExtractorPro(root)
    root.mainloop()
