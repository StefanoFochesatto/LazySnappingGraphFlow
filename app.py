import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import numpy as np
from PIL import Image, ImageTk
from tkinter import filedialog
import tkinter as tk
import numpy as np
import cv2
import maxflow
from sklearn.cluster import KMeans
import numpy as np
from matplotlib import pyplot as ppl
import networkx as nx


# Global variables
drawing = False
line_points_left_click = []
line_points_shift_left_click = []
source = set()  # Use a set to ensure unique tuples
sink = set()    # Use a set to ensure unique tuples


# Define computeDistFromSource function
def computeDistFromSource(dim, img, mean_colors):
    """
    Compute the minimum distance from each pixel to the set of mean colors.
    """
    DiffTensor = np.zeros((dim[0], dim[1], len(mean_colors)))
    for i in range(len(mean_colors)):
        # Ensure mean_colors[i] is a NumPy array
        mean_color = np.array(mean_colors[i], dtype=np.float32)

        # Subtract the mean color from the image using broadcasting
        diff = img.astype(np.float32) - mean_color  # Shape: (height, width, 3)

        # Compute the Euclidean distance across the color channels
        diffnorm = np.linalg.norm(diff, axis=2)  # Shape: (height, width)

        # Store the distances in DiffTensor
        DiffTensor[:, :, i] = diffnorm

    # Compute the minimum distances across all mean colors
    min_values = np.min(DiffTensor, axis=2)  # Shape: (height, width)

    return min_values


def compute_weights(dim, img, orientation):
    CI = np.zeros((dim[0] + 2, dim[1] + 2, dim[2]))
    CJ = np.zeros((dim[0] + 2, dim[1] + 2, dim[2]))
    CI[1:dim[0]+1, 1:dim[1] + 1, :] = img

    # [row, columns]
    # [vertical shift, horizontal shift]
    if orientation == 1:  # up and to the left
        CJ[2:dim[0]+2, 2:dim[1] + 2, :] = img
    elif orientation == 2:  # up
        CJ[2:dim[0]+2, 1:dim[1] + 1, :] = img
    elif orientation == 3:  # up and to the right
        CJ[2:dim[0]+2, 0:dim[1], :] = img
    elif orientation == 4:  # To the left
        CJ[1:dim[0]+1, 2:dim[1] + 2, :] = img
    elif orientation == 6:  # To the right
        CJ[1:dim[0]+1, 0:dim[1], :] = img
    elif orientation == 7:  # Down and to the left
        CJ[0:dim[0], 2:dim[1] + 2, :] = img
    elif orientation == 8:  # down
        CJ[0:dim[0], 1:dim[1] + 1, :] = img
    elif orientation == 9:  # down and to the right
        CJ[0:dim[0], 0:dim[1], :] = img
    diff = CI - CJ
    diff = diff[1:dim[0]+1, 1:dim[1] + 1, :]
    weights = 1/(np.linalg.norm(diff, axis=2) - 1)

    return weights


def create_graph(dim, img, SourceWeights, SinkWeights):
    g = maxflow.Graph[float]()
    nodeids = g.add_grid_nodes(dim[0:2])

   # Edges point up and to the left
    weights = compute_weights(dim, img, 1)
    structure = np.zeros((3, 3))
    structure[0, 0] = 1
    g.add_grid_edges(nodeids, structure=structure,
                     weights=weights, symmetric=False)

    # Edges point up
    weights = compute_weights(dim, img, 2)
    structure = np.zeros((3, 3))
    structure[0, 1] = 1
    g.add_grid_edges(nodeids, structure=structure,
                     weights=weights, symmetric=False)

    # Edges point up and to the right
    weights = compute_weights(dim, img, 3)
    structure = np.zeros((3, 3))
    structure[0, 2] = 1
    g.add_grid_edges(nodeids, structure=structure,
                     weights=weights, symmetric=False)

    # Edges point to the left
    weights = compute_weights(dim, img, 4)
    structure = np.zeros((3, 3))
    structure[1, 0] = 1
    g.add_grid_edges(nodeids, structure=structure,
                     weights=weights, symmetric=False)

    # Edges point to the right
    weights = compute_weights(dim, img, 6)
    structure = np.zeros((3, 3))
    structure[1, 2] = 1
    g.add_grid_edges(nodeids, structure=structure,
                     weights=weights, symmetric=False)

    # Edges pointing down and to left
    weights = compute_weights(dim, img, 7)
    structure = np.zeros((3, 3))
    structure[2, 0] = 1
    g.add_grid_edges(nodeids, structure=structure,
                     weights=weights, symmetric=False)

  # Edges pointing down
    weights = compute_weights(dim, img, 8)
    structure = np.zeros((3, 3))
    structure[2, 1] = 1
    g.add_grid_edges(nodeids, structure=structure,
                     weights=weights, symmetric=False)

  # Edges pointing down and to the right
    weights = compute_weights(dim, img, 9)
    structure = np.zeros((3, 3))
    structure[2, 2] = 1
    g.add_grid_edges(nodeids, structure=structure,
                     weights=weights, symmetric=False)

    g.add_grid_tedges(nodeids, SourceWeights, SinkWeights)

    return nodeids, g


# Title of the app
st.title("Interactive Image Segmentation App")

# Sidebar configuration
st.sidebar.title("Configuration")

# Upload image
uploaded_file = st.file_uploader(
    "Upload an Image", type=["png", "jpg", "jpeg"])
if uploaded_file is None:
    st.warning("Please upload an image to proceed.")
    st.stop()

# Open and display the image
image = Image.open(uploaded_file).convert('RGB')  # Ensure RGB mode
st.image(image, caption='Uploaded Image', use_container_width=True)

# Resize image for canvas if necessary
resize_width = st.sidebar.slider(
    "Resize image width for annotation (pixels)", 100, 800, 600)
aspect_ratio = image.height / image.width
new_size = (resize_width, int(resize_width * aspect_ratio))
resized_image = image.resize(new_size)

# Convert the resized image to a NumPy array
image_array = np.array(resized_image)

# Verify the shape of the image array
# Should be (height, width, 3)
st.write(f"Image array shape: {image_array.shape}")

# Annotation mode
annotation_mode = st.sidebar.selectbox(
    "Annotation Mode", ["Source (Green)", "Sink (Red)"])

# Stroke color based on annotation mode
stroke_color = "#00FF00" if annotation_mode == "Source (Green)" else "#FF0000"

# Stroke width
stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)

# Create a canvas component
canvas_result = st_canvas(
    fill_color="rgba(0, 0, 0, 0)",  # No fill
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_image=resized_image,
    update_streamlit=True,
    height=new_size[1],
    width=new_size[0],
    drawing_mode="freedraw",
    key="canvas",
)

# Initialize sets for source and sink pixels
if 'source_pixels' not in st.session_state:
    st.session_state.source_pixels = set()
if 'sink_pixels' not in st.session_state:
    st.session_state.sink_pixels = set()

# Process the annotations
if canvas_result.image_data is not None:
    annotated_image = canvas_result.image_data
    # Convert the annotated image to a NumPy array
    mask = np.array(annotated_image)[:, :, :3]  # Ignore alpha channel

    # Depending on the annotation mode, extract the coordinates of the annotated pixels
    if annotation_mode == "Source (Green)":
        # Find pixels that are green in the annotated image
        source_annotation = (mask == [0, 255, 0]).all(axis=2)
        # Get the coordinates of these pixels
        coords = np.argwhere(source_annotation)
        # For each coordinate, add it to the source_pixels set
        for y, x in coords:
            st.session_state.source_pixels.add((x, y))
    else:
        # Find pixels that are red in the annotated image
        sink_annotation = (mask == [255, 0, 0]).all(axis=2)
        # Get the coordinates of these pixels
        coords = np.argwhere(sink_annotation)
        # For each coordinate, add it to the sink_pixels set
        for y, x in coords:
            st.session_state.sink_pixels.add((x, y))

    # Display the accumulated annotations
    # Create an empty image to display the accumulated annotations
    annotated_display = np.zeros_like(mask)
    # Mark source pixels in green
    for x, y in st.session_state.source_pixels:
        if 0 <= x < new_size[0] and 0 <= y < new_size[1]:
            annotated_display[y, x] = [0, 255, 0]  # Green
    # Mark sink pixels in red
    for x, y in st.session_state.sink_pixels:
        if 0 <= x < new_size[0] and 0 <= y < new_size[1]:
            annotated_display[y, x] = [255, 0, 0]  # Red
    st.image(annotated_display, caption='Accumulated Annotations',
             use_container_width=True)

# Segmentation button
if st.button("Run Segmentation"):
    if not st.session_state.source_pixels or not st.session_state.sink_pixels:
        st.error(
            "Please annotate both source and sink regions before running segmentation.")
    else:
        # Placeholder for your segmentation algorithm
        def segment_image(image_array, source_pixels, sink_pixels):
            """
                Implement your graph flow segmentation algorithm here.
            """
            # Run KMeans on colors of source and sink pixels
            Source_rgb_values = [image_array[y, x] for x, y in source_pixels]
            Sink_rgb_values = [image_array[y, x] for x, y in sink_pixels]

            # Perform KMeans clustering on source and sink colors
            Source_kmeans = KMeans(
                n_clusters=3, random_state=0).fit(Source_rgb_values)
            Sink_kmeans = KMeans(
                n_clusters=3, random_state=0).fit(Sink_rgb_values)

            SourceMeanColors = Source_kmeans.cluster_centers_
            SinkMeanColors = Sink_kmeans.cluster_centers_

            img_shape = image_array.shape
            dF = computeDistFromSource(
                img_shape, image_array, SourceMeanColors)
            dB = computeDistFromSource(img_shape, image_array, SinkMeanColors)

            # Compute weights for terminal edges of the graph
            epsilon = 1e-8  # Small value to avoid division by zero
            SourceWeights = dF / (dF + dB + epsilon)
            SinkWeights = dB / (dF + dB + epsilon)

            # Adjust weights based on source and sink pixels
            for i in source_pixels:
                ir = tuple(reversed(i))
                SourceWeights[ir] = 0
                SinkWeights[ir] = np.inf

            for i in sink_pixels:
                ir = tuple(reversed(i))
                SourceWeights[ir] = np.inf
                SinkWeights[ir] = 0

            # Initialize graph, run flow algorithm, and visualize results
            nodeids, g = create_graph(
                img_shape, image_array, SourceWeights, SinkWeights)
            g.maxflow()

            sgm = g.get_grid_segments(nodeids)
            sgmimg = np.int_(np.logical_not(sgm))
            ResultImg = (sgmimg * 255).astype(np.uint8)

            return ResultImg

        # Prepare the image array
        image_array = np.array(resized_image)
        # Run segmentation
        segmented_image = segment_image(
            image_array, st.session_state.source_pixels, st.session_state.sink_pixels)
        # Display the segmented image
        st.image(segmented_image, caption='Segmented Image',
                 use_container_width=True)

        # Optionally, reset the annotations
        if st.button("Reset Annotations"):
            st.session_state.source_pixels = set()
            st.session_state.sink_pixels = set()
else:
    st.info("Annotate the image and click 'Run Segmentation' to proceed.")
