import os
import cv2
import torch
import imageio
import open3d as o3d
import numpy as np
from PIL import Image
import open3d.cuda.pybind.t.pipelines.registration as treg
import time
from skimage import img_as_ubyte
import matplotlib.pyplot as plt
from skimage import io
import copy
import quaternion
#from icp import calculate_icp
from pytorch3d.transforms import matrix_to_quaternion,quaternion_to_matrix
from pytorch3d.structures import Pointclouds
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
from pytorch3d.renderer import (PerspectiveCameras,PointsRasterizationSettings,PointsRenderer,PointsRasterizer,AlphaCompositor)


def calculate_icp(pc1_path,pc2_path):
	source = o3d.t.io.read_point_cloud(pc1_path)
	target = o3d.t.io.read_point_cloud(pc2_path)
	source.estimate_normals()
	target.estimate_normals()
	voxel_sizes = o3d.utility.DoubleVector([0.1, 0.05, 0.025])

	criteria_list = [treg.ICPConvergenceCriteria(relative_fitness=0.0001,relative_rmse=0.0001,max_iteration=20),treg.ICPConvergenceCriteria(0.00001, 0.00001, 15),treg.ICPConvergenceCriteria(0.000001, 0.000001, 10)]
	max_correspondence_distances = o3d.utility.DoubleVector([0.3, 0.14, 0.07])

	init_source_to_target = o3d.core.Tensor.eye(4, o3d.core.Dtype.Float32)
	estimation = treg.TransformationEstimationPointToPlane()


	callback_after_iteration = lambda loss_log_map : print("Iteration Index: {}, Scale Index: {}, Scale Iteration Index: {}, Fitness: {}, Inlier RMSE: {},".format(
	    loss_log_map["iteration_index"].item(),
	    loss_log_map["scale_index"].item(),
	    loss_log_map["scale_iteration_index"].item(),
	    loss_log_map["fitness"].item(),
	    loss_log_map["inlier_rmse"].item()))
	    
	s = time.time()
	registration_ms_icp = treg.multi_scale_icp(source, target, voxel_sizes,criteria_list,max_correspondence_distances,init_source_to_target, estimation,callback_after_iteration)

	ms_icp_time = time.time() - s

	return registration_ms_icp.transformation.numpy()

def int_qua_calculator(source,target,interpolate_rate):
	icp_estimated_RT_np = calculate_icp(source,target)
	icp_estimated_RT = torch.from_numpy(icp_estimated_RT_np)
	qua1 = matrix_to_quaternion(icp_estimated_RT[:3,:3])

	w_1 = qua1.detach().cpu().numpy()[0]
	x_1 = qua1.detach().cpu().numpy()[1]
	y_1 = qua1.detach().cpu().numpy()[2]
	z_1 = qua1.detach().cpu().numpy()[3]

	qua1_np = np.quaternion(w_1,x_1,y_1,z_1)
	qua2_np = np.quaternion(1,0,0,0)

	quat_interp = quaternion.slerp_evaluate(qua1_np, qua2_np, interpolate_rate)
	int_qua= quaternion.as_float_array(quat_interp)
	interpolated_RT = torch.eye(4)
	interpolated_RT[:3,:3] = quaternion_to_matrix(torch.from_numpy(int_qua))
	interpolated_RT[:3,3:] = icp_estimated_RT[:3,3:]/2
	return interpolated_RT


def combine_pointclouds(pc1,pc2):
    combined_pc_verts = np.vstack((np.asarray(pc1.points), np.asarray(pc2.points)))
    combined_pc_colors = np.vstack((np.asarray(pc1.colors), np.asarray(pc2.colors)))
    pcd = o3d.geometry.PointCloud()
    return o3d.utility.Vector3dVector(combined_pc_verts),o3d.utility.Vector3dVector(combined_pc_colors)

def draw_interpolated_pc(source, target):
    transformation = int_qua_calculator(source,target,0.5)
    pc1 = o3d.io.read_point_cloud(source)
    pc2 = o3d.io.read_point_cloud(target)
    pc1_temp = copy.deepcopy(pc1)
    pc2_temp = copy.deepcopy(pc2)
    pc1_temp.transform(transformation)
    transformation2 = np.eye(4)
    transformation2[:3,:3] = transformation[:3,:3].T
    transformation2[:3,3:] = -transformation[:3,3:]
    pc2_temp.transform(transformation2)
    pcd = o3d.geometry.PointCloud()
    pcd.points,pcd.colors = combine_pointclouds(pc1_temp,pc2_temp)
    o3d.visualization.draw_geometries([pcd])
    return torch.from_numpy(np.asarray(pcd.points)).cuda(),torch.from_numpy(np.asarray(pcd.colors)).cuda()


def render_rotated_frame(source, target,save_path,im_show=True):
    verts,colors = draw_interpolated_pc(source,target)
    min_val = torch.min(colors)
    max_val = torch.max(colors)
    colors = (colors - min_val) / (max_val - min_val)
    point_cloud = Pointclouds(points=[verts.float() ], features=[colors.float()])
    image_size = ((640, 480),)
    fcl_screen = ((525,525),)        
    prp_screen = ((240, 320),)
    cameras_ndc = PerspectiveCameras(device="cuda:0",focal_length=fcl_screen,principal_point=prp_screen,in_ndc=False,image_size=image_size)
    raster_settings = PointsRasterizationSettings(image_size=(480, 640),radius = 0.01,points_per_pixel = 10)
    rasterizer = PointsRasterizer(cameras=cameras_ndc, raster_settings=raster_settings).cuda()
    renderer = PointsRenderer(rasterizer=rasterizer,compositor=AlphaCompositor()).cuda()

    images = renderer(point_cloud)
    plt.figure(figsize=(10, 10))
    plt.imsave(save_path, (images[0, ..., :3].cpu().numpy()))
    if (im_show ==True):
    	plt.imshow(images[0, ..., :3].cpu().numpy())
    	plt.axis("off");
    	plt.show()
    
