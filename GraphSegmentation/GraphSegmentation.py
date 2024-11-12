from PIL import Image, ImageTk
from tkinter import filedialog
import tkinter as tk
import numpy as np
import cv2
import maxflow
from sklearn.cluster import KMeans
import numpy as np
from examples_utils import plot_graph_2d
from matplotlib import pyplot as ppl
import networkx as nx


# Global variables
drawing = False
line_points_left_click = []
line_points_shift_left_click = []
source = set()  # Use a set to ensure unique tuples
sink = set()    # Use a set to ensure unique tuples


def open_file_dialog():
    root = tk.Tk()
    root.withdraw()

    file_path = filedialog.askopenfilename(
        title="Select an Image", filetypes=[("All files", "*.*")])

    return file_path


def computeDistFromSource(dim, img, SourceMeanColors):

    DiffTensor = np.zeros((dim[0], dim[1], len(SourceMeanColors)))
    for i in range(len(SourceMeanColors)):
        KIF = np.ones(dim, dtype=np.uint8) * SourceMeanColors[i]
        diff = img - KIF
        diffnorm = np.linalg.norm(diff, axis=2)
        DiffTensor[:, :, i] = diffnorm

    # Find the indices of the minimum values along the chosen axis
    min_indices = np.argmin(DiffTensor, axis=2)

    # Use the indices to extract the minimum values along the chosen axis
    min_values = np.min(DiffTensor, axis=2)

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


# Mouse callback function

def draw_curve(event, x, y, flags, param):
    global drawing, line_points_left_click, line_points_shift_left_click, source, sink

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        if flags & cv2.EVENT_FLAG_SHIFTKEY:
            line_points_shift_left_click = [(x, y)]
        else:
            line_points_left_click = [(x, y)]

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if flags & cv2.EVENT_FLAG_SHIFTKEY:
            line_points_shift_left_click.append((x, y))
            draw_smooth_curve(img, line_points_shift_left_click, color=(
                0, 0, 255))  # Red curve for Shift + left click
            sink.update(get_pixels_in_line(line_points_shift_left_click))
        else:
            line_points_left_click.append((x, y))
            draw_smooth_curve(img, line_points_left_click, color=(
                0, 255, 0))  # Green curve for left click
            source.update(get_pixels_in_line(line_points_left_click))

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            if flags & cv2.EVENT_FLAG_SHIFTKEY:
                line_points_shift_left_click.append((x, y))
                draw_smooth_curve(img, line_points_shift_left_click, color=(
                    0, 0, 255))  # Red curve for Shift + left click
            else:
                line_points_left_click.append((x, y))
                draw_smooth_curve(img, line_points_left_click, color=(
                    0, 255, 0))  # Green curve for left click


def draw_smooth_curve(image, points, color):
    if len(points) > 1:
        for i in range(1, len(points)):
            cv2.line(image, points[i - 1], points[i], color, 2)
        cv2.imshow("Draw Curves", img)


def get_pixels_in_line(points):
    pixels = []
    for i in range(1, len(points)):
        line = np.linspace(points[i - 1], points[i], max(
            abs(points[i - 1][0] - points[i][0]), abs(points[i - 1][1] - points[i][1])))
        pixels.extend(line.astype(int))
    return set(map(tuple, pixels))  # Convert list to set of unique tuples


if __name__ == '__main__':
    image_path = open_file_dialog()
    # Create a black image
    OrigImg = cv2.imread(image_path)
    img = cv2.imread(image_path)

    cv2.imshow("Draw Curves", img)
    # Set the callback function for mouse events
    cv2.setMouseCallback("Draw Curves", draw_curve)
    while True:
        key = cv2.waitKey(1) & 0xFF
        # Press 'r' to reset the drawing
        if key == ord('r'):
            img = np.zeros((512, 512, 3), np.uint8)
            cv2.imshow("Draw Curves", img)
        # Press 'esc' to exit the program
        elif key == 27:
            break
    cv2.destroyAllWindows()


# Run kmeans on colors of source and sink
    Source_rgb_values = [OrigImg[y, x] for x, y in source]
    Sink_rgb_values = [OrigImg[y, x] for x, y in sink]

    Sourcekmeans = KMeans(n_clusters=3, random_state=0,
                          n_init="auto").fit(Source_rgb_values)

    Sinkkmeans = KMeans(n_clusters=3, random_state=0,
                        n_init="auto").fit(Sink_rgb_values)
    SourceMeanColors = Sourcekmeans.cluster_centers_.astype(int)
    SinkMeanColors = Sinkkmeans.cluster_centers_.astype(int)


# Compute Color Distances,
    dF = computeDistFromSource(img.shape, OrigImg, SourceMeanColors)
    dB = computeDistFromSource(img.shape, OrigImg, SinkMeanColors)

# Compute weights for tedges of graph
    SourceWeights = dF/(dF + dB)
    SinkWeights = dB/(dF + dB)

    for i in source:
        ir = tuple(reversed(i))
        SourceWeights[ir] = 0
        SinkWeights[ir] = np.inf

    for i in sink:
        ir = tuple(reversed(i))
        SourceWeights[ir] = np.inf
        SinkWeights[ir] = 0


# initilize graph, run flow algo, and visualize results
    nodeids, g = create_graph(
        img.shape, OrigImg, SourceWeights, SinkWeights)
    g.maxflow()

   # net = Network()
   # net.from_nx(g.get_nx_graph())
   # net.save_graph("networkx-pyvis.html")
    # HTML(filename="networkx-pyvis.html")

    #
    # plot_graph_2d(g, nodeids.shape)

    sgm = g.get_grid_segments(nodeids)
    sgmimg = np.int_(np.logical_not(sgm))
    ResultImg = (sgmimg * 255).astype(np.uint8)

    cv2.namedWindow("Results", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Original", cv2.WINDOW_NORMAL)

    cv2.imshow("Results", ResultImg)
    cv2.imshow("Original", OrigImg)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
