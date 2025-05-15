# Region Creator Tool Documentation

## Overview

The Region Creator is a companion tool for the Guideway object detection system. It provides a graphical interface for creating and managing regions of interest (ROIs) that will be used by the main application for targeted object detection.

## Purpose

The primary purpose of this tool is to allow users to:

1. Create and manage multiple mask models
2. Define polygon regions over a webcam image
3. Save these regions to a JSON file (`regions.json`)
4. Provide an intuitive interface for precise region definition

## Key Features

### Mask Model Management

- **Create Models**: Define different sets of regions for different detection scenarios
- **Rename Models**: Update model names as needed
- **Delete Models**: Remove unwanted models

### Region Creation

- **Rectangle Regions**: Draw rectangular regions by clicking and dragging
- **Polygon Regions**: Define precise polygon regions by placing points
- **Visual Feedback**: Semi-transparent overlays show region coverage

### Editing Tools

- **Undo/Redo**: Full undo/redo functionality for all operations
- **Region Selection**: Click on regions to select them for editing or deletion
- **Keyboard Shortcuts**: Convenient shortcuts for common operations

## Technical Implementation

### Data Structure

The region definitions are stored in a JSON file with the following structure:

```json
{
  "model_name": {
    "regions": {
      "region-1": [[x1, y1], [x2, y2], [x3, y3], ...],
      "region-2": [[x1, y1], [x2, y2], [x3, y3], ...],
      ...
    }
  },
  "another_model": {
    "regions": {
      ...
    }
  }
}
```

Each model contains a dictionary of regions, and each region is defined by a list of coordinate points.

### Camera Handling

The tool captures a single reference frame from the webcam for region definition. This approach provides several benefits:

1. Consistent reference for precise region placement
2. Reduced processing requirements compared to live video
3. Better visibility of region boundaries

A camera warm-up period ensures proper exposure and white balance before capturing the reference frame.

### User Interface

The interface is divided into two main sections:

1. **Control Panel**: Contains all controls for model and region management
2. **Canvas**: Displays the webcam image and allows for region drawing and selection

The UI uses the modern Sun Valley theme (sv_ttk) for a clean, professional appearance.

## Usage Guide

### Creating a New Mask Model

1. Click the "New Mask..." button
2. Enter a name for the model
3. Click OK

### Adding Regions

1. Select a mask model from the dropdown
2. Click the "Add Region" button (or press 'N')
3. Click and drag on the canvas to create a rectangular region
4. Release the mouse button to complete the region

### Selecting and Deleting Regions

1. Click on a region in the listbox or directly on the canvas
2. The selected region will be highlighted
3. Click "Delete Region" (or press Delete) to remove it

### Saving Changes

Changes are automatically saved to the `regions.json` file after each operation. You can also click "Save and Close" to explicitly save and exit the application.

## Integration with Main Application

The regions defined with this tool are used by the Guideway application to:

1. Limit object detection to specific areas of the camera view
2. Reduce false positives by ignoring objects outside defined regions
3. Optimize processing by focusing only on areas of interest

## Keyboard Shortcuts

- **Ctrl+Z**: Undo the last operation
- **Ctrl+Y**: Redo the last undone operation
- **N**: Create a new region
- **Delete**: Delete the selected region

## Best Practices

1. **Use Clear Model Names**: Choose descriptive names for your mask models
2. **Define Precise Regions**: Take time to accurately define regions around areas of interest
3. **Use Multiple Models**: Create different models for different detection scenarios
4. **Regular Backups**: Although changes are saved automatically, consider backing up your `regions.json` file periodically
