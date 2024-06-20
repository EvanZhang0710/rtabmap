import os
import numpy as np
import copy
import open3d as o3d



def crop(file_path, save_path):
    print(f"Demo for manual geometry cropping: {file_path}")
    print(
        "1) Press 'Y' twice to align geometry with negative direction of y-axis"
    )
    print("2) Press 'K' to lock screen and to switch to selection mode")
    print("3) Drag for rectangle selection,")
    print("   or use ctrl + left click for polygon selection")
    print("4) Press 'C' to get a selected geometry and to save it")
    print("5) Press 'F' to switch to freeview mode")

    mesh = o3d.io.read_triangle_mesh(file_path)
    if mesh.is_empty():
        print(f"Warning: The mesh file {file_path} is empty or cannot be read.")
        return None

    pcd = mesh.sample_points_uniformly(number_of_points=100000)
    if pcd.is_empty():
        print(f"Warning: The point cloud from mesh {file_path} is empty.")
        return None

    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()

    cropped_pcd = vis.get_cropped_geometry()
    if cropped_pcd is None or cropped_pcd.is_empty():
        print("Warning: The cropped point cloud is empty or cropping was not performed.")
        return None

    # Save the cropped point cloud
    # base_name = os.path.basename(file_path).replace('.obj', '_cropped.obj')
    # save_file_path = os.path.join(save_path, base_name)
    # o3d.io.write_point_cloud(save_file_path, cropped_pcd)
    # print(f"Cropped point cloud saved to: {save_file_path}")

    return cropped_pcd


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])


def pick_points(pcd):
    print("")
    print(
        "1) Please pick at least three correspondences using [shift + left click]"
    )
    print("   Press [shift + right click] to undo point picking")
    print("2) After picking points, press q to close the window")
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()
    print("")
    return vis.get_picked_points()


def manual_registration(source, target, num):
    print("Demo for manual ICP")

    if source.is_empty() or target.is_empty():
        print("Warning: One of the point clouds is empty after sampling.")
        return None

    print("Visualization of two point clouds before manual alignment")
    draw_registration_result(source, target, np.identity(4))

    picked_id_source = pick_points(source)
    picked_id_target = pick_points(target)

    if len(picked_id_source) < num or len(picked_id_target) < num:
        print("Error: At least three correspondences are required.")
        return None
    if len(picked_id_source) != len(picked_id_target):
        print("Error: The number of picked points do not match.")
        return None

    # Convert the picked points to numpy array
    corr = np.zeros((len(picked_id_source), 2))
    corr[:, 0] = picked_id_source
    corr[:, 1] = picked_id_target

    # estimate rough transformation using correspondences
    print("Compute a rough transform using the correspondences given by user")
    p2p = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    trans_init = p2p.compute_transformation(source, target,
                                            o3d.utility.Vector2iVector(corr))

    # point-to-point ICP for refinement
    print("Perform point-to-point/point-to-plane ICP refinement")
    threshold = 0.03 # 3cm distance threshold
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    draw_registration_result(source, target, reg_p2p.transformation)
    return reg_p2p.transformation


if __name__ == "__main__":

    # Directory path for obj files
    dir_path = "/Users/zhangyiwenevan/Desktop/bure/Scans/test1/"
    # Directory path to save cropped obj files
    save_path = "/Users/zhangyiwenevan/Desktop/bure/Scans/cropped objs/"
    # Minimum number of points to be picked for manual registration 
    min_pts_picked = 3 
    
    os.makedirs(save_path, exist_ok=True)

    obj_files = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith('.obj')]

    # Storing the cropped point clouds
    cropped_files = []

    # Process cropping for each file and save cropped files
    for obj_file in obj_files:
        cropped_pcd = crop(obj_file, save_path)
        if cropped_pcd is not None and not cropped_pcd.is_empty():
            cropped_files.append(cropped_pcd)

    # Perform manual registration iteratively
    if cropped_files:
        cumulative_pcd = cropped_files[0]
        for i in range(1, len(cropped_files)):
            target_pcd = cropped_files[i]
            transformation = manual_registration(cumulative_pcd, target_pcd, min_pts_picked)
            if transformation is not None:
                cumulative_pcd.transform(transformation)
                cumulative_pcd += target_pcd  # Merge the aligned point cloud into the cumulative point cloud
                print(f"Aligned and merged point cloud {i + 1}")
