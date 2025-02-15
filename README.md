# Map Generation - Interactive Goldberg Polyhedron

This project generates and visualizes an interactive 3D Goldberg polyhedron (a sphere-like shape made up of hexagons and pentagons) using Python. The visualization includes features like face selection, zooming, and dynamic information display.

## Features

- Generate icospheres with customizable subdivision levels
- Create Goldberg polyhedrons (similar to geodesic domes)
- Interactive 3D visualization
- Click to select individual faces
- Zoom functionality
- Real-time face information display
- Organic appearance with customizable noise levels

## Requirements

- Python 3.8 or higher
- NumPy
- Matplotlib

## Installation

1. Clone this repository:
```bash
git clone [your-repository-url]
cd map-generation
```

2. Create and activate a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

Run the main script with:
```bash
python icosphere.py
```

### Interaction Controls:
- **Left Click**: Select/deselect faces
- **Scroll Wheel**: Zoom in/out
- **Click and Drag**: Rotate the view

### Customization

You can modify the following parameters in `icosphere.py`:
- `subdivision_level`: Controls the resolution of the sphere (higher values create more faces)
- `noise_level`: Controls the organic appearance (higher values create more irregular surfaces)

Example with custom parameters:
```python
plot_goldberg(subdivision_level=2, noise_level=0.03)
```