import cv2
import numpy as np
import os
import json
from datetime import datetime
from argparse import ArgumentParser
import pyk4a
from pyk4a import Config, ImageFormat, PyK4A, PyK4ARecord, WiredSyncMode, connected_device_count
import sys
import pykinect_azure as pykinect
import threading 




def capture_and_save(devices, reference_serial='000753613012'):
    #7*10 100mm
    board_width = 9 # 棋盘格内部角点的行数
    board_height = 6 # 棋盘格内部角点的列数
    square_size = 100 # 每个格子的实际大小（单位：mm）
    obj_points = {} # 3D点
    img_points = {} # 2D点
    for device in devices:
        device.start()
        
        objp = np.zeros((board_height * board_width, 3), np.float32)
        objp[:, :2] = np.mgrid[0:board_width, 0:board_height].T.reshape(-1, 2) * square_size
        id = device.serial
        img_points[id] = []
        obj_points[id] = []
        # while True:
        #     capture = device.get_capture()
        #     if np.any(capture.color):
        #         cv2.imshow(f"k4a+{id}", capture.color[:, :, :3])
        #         key = cv2.waitKey(10)
        #         if key != -1:
        #             cv2.destroyAllWindows()
        #             break
        # capture_count = 0
        # file_name = f"{id}_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}"
        # os.makedirs(os.path.join("C:/Users/Dell/Desktop/yy_bev/calibration_temp", file_name),exist_ok=True)
        # file_name = os.path.join("C:/Users/Dell/Desktop/yy_bev/calibration_temp", file_name)
        # while capture_count < 10: # 至少捕获10张棋盘格图像/
        capture = device.get_capture()
        color_image = capture.color[:,:,:3]
        # color_image = cv2.imdecode(np.frombuffer(image, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        # print(color_image.shape)
        # cv2.imshow("Captured Image", color_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        if id == reference_serial:
            gray_ref = gray
        else: 
            id_sub = id
            gray_sub = gray
        ret, corners = cv2.findChessboardCorners(gray, (board_width, board_height), None)
        if ret:
            
            img_points[id].append(corners)
            obj_points[id].append(objp)
            # capture_count += 1

    # 显示角点
            cv2.drawChessboardCorners(gray, (board_width, board_height), corners, ret)
            # cv2.imshow(f"Device {device.serial}", color_image)
            # cv2.waitKey(500) # 显示500ms
            # image_file = os.path.join(file_name, f"image_{capture_count+1}.png")
            # cv2.imwrite(image_file, gray)
            # print(f"保存图像{image_file}")
        else:
            print(f"未能在设备 {id} 中检测到角点")
            exit()
        print(f"设备 {id} 捕获完成")

    print('开始进行参考相机标定')
    ret, mtx_ref, dist_ref, rvecs_ref, tvecs_ref = cv2.calibrateCamera(obj_points[reference_serial], img_points[reference_serial], gray_ref.shape[::-1], None, None)
    print('开始进行从机的标定')
    ret, mtx_sub, dist_sub, rvecs_sub, tvecs_sub = cv2.calibrateCamera(obj_points[id_sub], img_points[id_sub], gray_sub.shape[::-1], None, None)

    #print("开始标定多相机...")
    #ret, mtx1, dist1, mtx2, dist2, R, T, E, F = cv2.stereoCalibrate(obj_points[reference_serial], img_points[reference_serial], img_points[id_sub], mtx_ref, dist_ref, mtx_sub, dist_sub, gray_ref.shape[::-1], criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5))
    result = {}
    result[reference_serial]={
            'mtx': mtx_ref.tolist(),
            'dist': dist_ref.tolist(),
            'R': rvecs_ref[0].flatten().tolist(),
            'T': tvecs_ref[0].flatten().tolist()
        }

    result[id_sub]={
            'mtx': mtx_sub.tolist(),
            'dist': dist_sub.tolist(),
            # 'R': R[0].flatten().tolist(),
            # 'T': T.flatten().tolist()
            'R': rvecs_sub[0].flatten().tolist(),
            'T': tvecs_sub[0].flatten().tolist()
        }
    output_file = f'clibs.json'
    with open(output_file, 'w')as f:
        json.dump(result, f, indent=4)



cnt = connected_device_count()
if not cnt:
    print("No devices available")
    exit()
print(f"Available devices: {cnt}")

devices_dic = {'slave': []}
for device_id in range(cnt):
    device = PyK4A(device_id=device_id)
    device.open()
    print(f"{device_id}: {device.serial}")
    jack_in, jack_out = device.sync_jack_status
    if (not jack_in) and (not jack_out):
        print("No jack connected")
        exit()
    else:
        if (not jack_in) and jack_out:
            devices_dic['master'] = device_id
        else:
            devices_dic['slave'].append(device_id)
    device.close()



resolution = pyk4a.ColorResolution.RES_720P
fps = pyk4a.FPS.FPS_30


devices = []
configs = []

config = Config(wired_sync_mode=WiredSyncMode.MASTER, 
                color_resolution=resolution, 
                camera_fps=fps,
                # color_format=ImageFormat.COLOR_MJPG,
                depth_mode=pyk4a.DepthMode.NFOV_2X2BINNED)

configs.append(config)
devices.append(PyK4A(config=config, device_id=devices_dic['master']))

delay_count = 1
for device_id in devices_dic['slave']:
    config = Config(wired_sync_mode=WiredSyncMode.SUBORDINATE,
                    color_resolution=resolution, 
                    camera_fps=fps,
                    # color_format=ImageFormat.COLOR_MJPG,
                    depth_mode=pyk4a.DepthMode.NFOV_2X2BINNED,
                    subordinate_delay_off_master_usec=160*delay_count)
    devices.append(PyK4A(config=config, device_id=device_id))
    delay_count += 1
    configs.append(config)
    

print("Recording... Press CTRL-C to stop recording.")
# task_list = []
# for device in devices:
#     device.start()
    # capture = devices[i].get_capture()
    # record_list[i].write_capture(capture)
# t = threading.Thread(target=capture_and_save, args=(devices,))
# t.start()
# task_list.append(t)
# # break
# for t in task_list:
#     t.join()
# for device in devices:
#     device.start()
#     capture_and_save(device=device)
capture_and_save(devices)


    


for device in devices:
    id = device.serial
    device.close()
    print(f"设备 {id} 已关闭")

cv2.destroyAllWindows()
