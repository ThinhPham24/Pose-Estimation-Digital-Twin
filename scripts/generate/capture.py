# import os
# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from scipy.spatial import distance_matrix
# import vedo

# """
# Author： Wanqing Xia
# Email: wxia612@aucklanduni.ac.nz

# This is the script to calculate the min, mean, max angular distance between camera points sampled 
# on a sphere surrounding the object, helps us to determine the sampling density
# """


# def fibonacci_sphere(samples=1, radius=1):
#     """
#     Generates points on the surface of a sphere using the Fibonacci method.
#     :param samples: Number of points to generate
#     :param radius: Radius of the sphere
#     :return: List of points on the sphere surface
#     """
#     points = []
#     phi = np.pi * (3. - np.sqrt(5.))  # golden angle in radians

#     for i in range(samples):
#         y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
#         radius_at_y = np.sqrt(1 - y * y) * radius  # radius at y, scaled by the desired radius

#         theta = phi * i  # golden angle increment

#         x = np.cos(theta) * radius_at_y
#         z = np.sin(theta) * radius_at_y
#         y *= radius  # scale y coordinate by the desired radius

#         points.append((x, y, z))

#     return points


# def angular_distance(point1, point2):
#     """
#     Calculate the angular distance in degrees between two points on a sphere.
#     """
#     inner_product = np.dot(point1, point2) / (np.linalg.norm(point1) * np.linalg.norm(point2))
#     angle_rad = np.arccos(np.clip(inner_product, -1.0, 1.0))
#     return np.degrees(angle_rad)


# if __name__ == "__main__":
#     # Generate 4000 points
#     radius = 5
#     points = fibonacci_sphere(42, radius)  # radius is 1 for simplicity

#     # Calculate distance matrix
#     dist_matrix = distance_matrix(points, points)

#     # Sort each row in the distance matrix and take the distances to the 5 nearest neighbors
#     nearest_dists = np.sort(dist_matrix, axis=1)[:, 1:6]

#     # Calculate the angular distances for each point to its 5 nearest neighbors
#     angular_dists = []
#     for i in range(len(points)):
#         for j in range(5):
#             neighbor_idx = np.where(dist_matrix[i] == nearest_dists[i, j])[0][0]
#             angular_dists.append(angular_distance(points[i], points[neighbor_idx]))

#     # Calculate min, max, and mean of the angular distances
#     min_angular_dist = np.min(angular_dists)
#     max_angular_dist = np.max(angular_dists)
#     mean_angular_dist = np.mean(angular_dists)
#     print("min angular distance: ", min_angular_dist)
#     print("max angular distance:", max_angular_dist)
#     print("mean angular distance: ", mean_angular_dist)

#     # Load 3D model using vedo
#     model_path = '/home/airlab/Desktop/DigitalTwin_PoseEstimation/data/ply_models/obj1.ply'
#     # texture_path = '/media/iai-lab/wanqing/YCB_Video_Dataset/models/035_power_drill/texture_map.png'

#     # Load the model
#     model = vedo.load(model_path)

#     # Load the texture
#     # texture = vedo.load(texture_path)

#     # Apply the texture manually
#     # model.texture(texture)

#     # Create a Plotter, add the model, and display
#     plot = vedo.Plotter()
#     plot.add(model)
#     model.scale(50)  # adjust scaling to fit your scene

#     # Define axis lines (simple)
#     x_line = vedo.shapes.Line([-5, 0, 0], [5, 0, 0], c='red')  # Red line for the X-axis
#     y_line = vedo.shapes.Line([0, -5, 0], [0, 5, 0], c='green')  # Green line for the Y-axis
#     z_line = vedo.shapes.Line([0, 0, -5], [0, 0, 5], c='blue')  # Blue line for the Z-axis

#     # Add the axis lines to the plot
#     plot.add(x_line)
#     plot.add(y_line)
#     plot.add(z_line)

#     # Add points
#     for point in points:
#         plot += vedo.shapes.Sphere(pos=point, r=0.1, c='b')
#     plot.show()
import os
import numpy as np
import vedo
import vedo
from vedo import Plotter, Ellipsoid

model_path = '/home/airlab/Desktop/DigitalTwin_PoseEstimation/data/ply_models/obj1.ply'
mesh = vedo.load(model_path) #load the ply as a mesh.

# emsh = Ellipsoid().scale(0.04).pos(0.028, 0.015, 0.015).wireframe()

# cut_mesh = mesh.clone().cut_with_mesh(emsh).c("gold").bc("t")
cut_mesh = mesh
# # Convert the cut mesh to a point cloud
# points = cut_mesh.points 
# point_cloud = vedo.Points(points, c="gold") # Create a vedo Points object

# # Save the point cloud as a PLY file
# output_path = "cut_mesh_pointcloud.ply"
# point_cloud.write(output_path)

# plt = Plotter(N=2, axes=1)
# plt.at(0).show(mesh, emsh, "Original Mesh and Cutting Ellipsoid")
# plt.at(1).show(cut_mesh, viewup="z", title="Mesh Cut with Ellipsoid")
# plt.interactive().close()
# Define the view parameters for each subplot
views = [
    
    {"cam": {"viewup": "y", "azimuth": 90, "elevation": 0}, "title": "Left View"}, # Left
    {"cam": {"viewup": "y", "azimuth": -90, "elevation": 0}, "title": "Right View"}, #Right
    {"cam": {"viewup": "y", "azimuth": 0, "elevation": 0}, "title": "Front View"}, #Front
    {"cam": {"viewup": "y", "azimuth": 180, "elevation": 0}, "title": "Back View"}, #Back
    {"cam": {"viewup": "z", "azimuth": 0, "elevation": 90}, "title": "Top View"},  # Top
]

plt = Plotter(N=5, axes=1)

for i, view in enumerate(views):
    cam = view["cam"]
    azimuth = cam["azimuth"]
    elevation = cam["elevation"]
    elevation_rad = np.radians(90 - elevation)
    azimuth_rad = np.radians(azimuth)
    camera_direction = np.array([
        np.sin(elevation_rad) * np.cos(azimuth_rad),
        np.sin(elevation_rad) * np.sin(azimuth_rad),
        np.cos(elevation_rad),
    ])

    # Filter visible points
    visible_points = []
    for point in cut_mesh.points:
        vector_to_point = point - cut_mesh.center_of_mass()
        if np.dot(vector_to_point, camera_direction) >= 0:
            visible_points.append(point)

    visible_point_cloud = vedo.Points(visible_points, c="gold")

    plt.at(i).show(visible_point_cloud, viewup=cam["viewup"], azimuth=azimuth, elevation=elevation, title=view["title"])

plt.interactive().close()

# Save the visible points from the first view to a PLY file.
elevation_rad = np.radians(0)
azimuth_rad = np.radians(0)
camera_direction = np.array([
    np.sin(elevation_rad) * np.cos(azimuth_rad),
    np.sin(elevation_rad) * np.sin(azimuth_rad),
    np.cos(elevation_rad),
])
visible_points = []
for point in cut_mesh.points:
    vector_to_point = point - cut_mesh.center_of_mass()
    if np.dot(vector_to_point, camera_direction) >= 0:
        visible_points.append(point)

visible_point_cloud = vedo.Points(visible_points, c="gold")
output_path = "visible_points.ply"
visible_point_cloud.write(output_path)
print(f"Visible points from top view saved to: {output_path}")


# def setup_scene(model_path, radius=5):
#     """
#     Load the 3D model and set up the visualization scene.
#     """
#     model = vedo.load(model_path)
#     model.scale(50)  # Scale the model appropriately

#     # Define axis lines
#     x_line = vedo.shapes.Line([-radius, 0, 0], [radius, 0, 0], c='red')  # X-axis
#     y_line = vedo.shapes.Line([0, -radius, 0], [0, radius, 0], c='green')  # Y-axis
#     z_line = vedo.shapes.Line([0, 0, -radius], [0, 0, radius], c='blue')  # Z-axis
    
#     return model, [x_line, y_line, z_line]

# # Define camera positions for capturing point clouds
# camera_positions = {
#     "top": (0, 0, 5),
#     "left": (-5, 0, 0),
#     "right": (5, 0, 0),
#     "front": (0, -5, 0),
#     "back": (0, 5, 0),
# }

# def capture_partial_point_cloud(model_path, output_folder):
#     """
#     Captures partial point clouds from different views and saves them.
#     """
#     os.makedirs(output_folder, exist_ok=True)
#     model, axes = setup_scene(model_path)

#     for view, pos in camera_positions.items():
#         plot = vedo.Plotter()
#         plot.add(model)
#         plot.add(*axes)

#         # Set camera position and render the view
#         plot.camera.SetPosition(pos)
#         print("THIONH")
#         plot.show()

#         # # Capture point cloud from the model
#         print("mode", model)
#         point_cloud = model.mesh()
#         print("mode", point_cloud)

#         # # Save the point cloud to a file
#         # np.savetxt(os.path.join(output_folder, f"{view}_point_cloud.txt"), point_cloud)
#         # print(f"Saved {view} point cloud")

#         plot.close()

# if __name__ == "__main__":
#     model_path = '/home/airlab/Desktop/DigitalTwin_PoseEstimation/data/ply_models/obj1.ply'
#     output_folder = "partial_point_clouds"
#     capture_partial_point_cloud(model_path, output_folder)

