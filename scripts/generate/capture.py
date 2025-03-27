import os
import numpy as np
import vedo
import open3d as o3d
from vedo import Plotter

class CADViewGenerator:
    def __init__(self, cad_model_path, output_dir):
        self.cad_model_path = cad_model_path
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.mesh = vedo.load(cad_model_path)

    def _write_ply(self, fn, verts):
        """Writes a point cloud to a PLY file."""
        verts = verts.reshape(-1, 3)
        ply_header = '''ply
        format ascii 1.0
        element vertex %(vert_num)d
        property float x
        property float y
        property float z
        end_header
        '''
        with open(fn, 'wb') as f:
            f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
            np.savetxt(f, verts, fmt='%f %f %f')

    def capture_cad_views_with_added_color(self):
        """Captures and saves specified views of a 3D CAD model with added color."""
        self.mesh.color = np.tile([200, 100, 255], (self.mesh.npoints, 1))

        views = [
            {"cam": {"viewup": [0.1, 0.0, 0.1]}, "azimuth": 0, "elevation": 0, "title": "Top"},
            {"cam": {"viewup": [0.0002, 0.0002, 0]}, "azimuth": 0, "elevation": 360 - 45, "title": "left_top"},
            {"cam": {"viewup": [-0.0002, 0.0002, 0]}, "azimuth": 0, "elevation": 360 - 45, "title": "right_top"},
            {"cam": {"viewup": [0.0002, -0.0002, 0]}, "azimuth": 0, "elevation": 360 - 45, "title": "front_top"},
            {"cam": {"viewup": [-0.0002, -0.0002, 0]}, "azimuth": 0, "elevation": 360 - 45, "title": "back_top"},
        ]

        for i, view in enumerate(views):
            plt = Plotter(offscreen=True)
            plt.show(self.mesh, camera=dict(pos=(0, 0, 0.4), viewup=view["cam"]["viewup"]), azimuth=view["azimuth"], elevation=view["elevation"])
            plt.screenshot(os.path.join(self.output_dir, f"{view['title']}.png"))
            plt.close()

    def generate_partial_point_clouds(self):
        """Generates partial point clouds from different views of a CAD model."""
        views = [
            {"cam": {"viewup": "y", "azimuth": 90, "elevation": 0}, "title": "left_view"},
            {"cam": {"viewup": "y", "azimuth": -90, "elevation": 0}, "title": "right_view"},
            {"cam": {"viewup": "y", "azimuth": 0, "elevation": 0}, "title": "front_view"},
            {"cam": {"viewup": "y", "azimuth": 180, "elevation": 0}, "title": "back_view"},
            {"cam": {"viewup": "z", "azimuth": 0, "elevation": 90}, "title": "top_view"},
        ]

        plt = Plotter(N=len(views), axes=1)

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

            visible_points = []
            for point in self.mesh.points:
                vector_to_point = point - self.mesh.center_of_mass() + 0.005
                if np.dot(vector_to_point, camera_direction) >= 0:
                    visible_points.append(point)

            points = np.asarray(visible_points)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=10))
            pcd.orient_normals_consistent_tangent_plane(k=5)
            output_path = os.path.join(self.output_dir, f"{view['title']}.ply")
            o3d.io.write_point_cloud(output_path, pcd, write_ascii=True)

            visible_point_cloud = vedo.Points(visible_points, c="gold")
            plt.at(i).show(visible_point_cloud, viewup=cam["viewup"], azimuth=azimuth, elevation=elevation, title=view["title"])

        plt.close()
        plt.interactive().close()

# if __name__ == "__main__":
#     cad_model_path = '/home/airlab/Desktop/DigitalTwin_PoseEstimation/data/ply_models/obj1.ply'
#     output_dir = "/home/airlab/Desktop/DigitalTwin_PoseEstimation/scripts/partial_point_clouds"

#     generator = CADViewGenerator(cad_model_path, output_dir)
#     generator.generate_partial_point_clouds()
#     print(f"Partial point clouds saved to {output_dir}")

#     generator.capture_cad_views_with_added_color()
#     print(f"CAD views saved to {output_dir}")

