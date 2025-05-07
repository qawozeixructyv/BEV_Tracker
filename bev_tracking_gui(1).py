import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QCheckBox, QScrollArea
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer
from detection import detect_people, project_to_bev
import os
import json
import pyk4a
from pyk4a import Config, ImageFormat, PyK4A, PyK4ARecord, WiredSyncMode, connected_device_count
import time

# ===== 卡尔曼滤波器初始化 =====
kalman = cv2.KalmanFilter(4, 2)
kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
kalman.statePre = np.zeros((4, 1), dtype=np.float32)


class CameraWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Multi-Camera BEV Detection UI")
        self.resize(1400, 800)
        self.trajectories = []
        self.BEV_SIZE = 500
        self.cameras = []
        self.checkboxes = []
        self.kinects = []
        self.labels = []
        self.bev_label = QLabel("BEV 轨迹图")
        self.bev_canvas = np.zeros((self.BEV_SIZE, self.BEV_SIZE, 3), dtype=np.uint8) # 初始化 BEV 画布

        self.layout = QHBoxLayout()
        self.setLayout(self.layout)

        #初始化设备列表
        cnt = connected_device_count()
        if not cnt:
            print("No devices available")
            exit()
        print(f"Available devices: {cnt}")

        self.devices_dic = {'slave': []}
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
                    self.devices_dic['master'] = device_id
                else:
                    self.devices_dic['slave'].append(device_id)
            device.close()

        resolution = pyk4a.ColorResolution.RES_720P
        fps = pyk4a.FPS.FPS_30


        self.devices = []
        configs = []

        config = Config(wired_sync_mode=WiredSyncMode.MASTER, 
                        color_resolution=resolution, 
                        camera_fps=fps,
                        # color_format=ImageFormat.COLOR_MJPG,
                        depth_mode=pyk4a.DepthMode.NFOV_2X2BINNED)

        configs.append(config)
        self.devices.append(PyK4A(config=config, device_id=self.devices_dic['master']))

        delay_count = 1
        for device_id in self.devices_dic['slave']:
            config = Config(wired_sync_mode=WiredSyncMode.SUBORDINATE,
                            color_resolution=resolution, 
                            camera_fps=fps,
                            # color_format=ImageFormat.COLOR_MJPG,
                            depth_mode=pyk4a.DepthMode.NFOV_2X2BINNED,
                            subordinate_delay_off_master_usec=160*delay_count)
            self.devices.append(PyK4A(config=config, device_id=device_id))
            delay_count += 1
            configs.append(config)

        # 左侧：相机选择
        self.cam_selector = QVBoxLayout()
        self.start_button = QPushButton("开始检测")
        self.start_button.clicked.connect(self.start_detection)
        self.cam_selector.addWidget(self.start_button)

        for i in range(connected_device_count()):
            cb = QCheckBox(f"Camera {i}")
            self.checkboxes.append(cb)
            self.cam_selector.addWidget(cb)

        left_panel = QWidget()
        left_panel.setLayout(self.cam_selector)
        self.layout.addWidget(left_panel)

        # 中间：相机画面 + BEV
        self.display_area = QVBoxLayout()

        scroll_widget = QWidget()
        self.scroll_layout = QVBoxLayout()
        scroll_widget.setLayout(self.scroll_layout)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(scroll_widget)

        self.display_area.addWidget(scroll)
        self.display_area.addWidget(self.bev_label)

        self.layout.addLayout(self.display_area)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frames)

        #读取相机参数
        root_path = 'C:/Users/Dell/Desktop/yy_bev'
        file_name = 'clibs.json'
    
        with open(os.path.join(root_path, file_name), 'r') as f:
            calib_data = json.load(f)

        self.camera_params = {}

        for cam_id, params in calib_data.items():
            print(f'相机:{cam_id}')

            k = np.array(params['mtx'])
            dist = np.array(params['dist'])
            revec = np.array(params['R']).reshape(3,1)
            tevec = np.array(params['T']).reshape(3,1)

            self.camera_params[cam_id] = {
                "internal params": k,
                "dist_coeffs": dist,
                "rotation_matrix": revec,
                "translation_vector": tevec
            }

    def start_detection(self):
        for cb, device in zip(self.checkboxes, self.devices):
            if cb.isChecked():
                # kinect = PyK4A(device_id=i, config=Config())
                # kinect.start()
                # self.kinects.append(kinect)
                device.start()
                id = device.serial

                label = QLabel(f"Camera {id} Preview")
                self.labels.append(label)
                self.scroll_layout.addWidget(label)

        self.timer.start(100)

    def update_frames(self):
        all_bev_points = []

        for i, kinect in enumerate(self.devices):
            id = kinect.serial
            print(id)
            mtx, _,  rotation_vector, translation_vector = self.camera_params[id].values()

            capture = kinect.get_capture()
            if capture.color is None:
                continue

            img = capture.color[:,:,:3]
            people_boxes, display_img = detect_people(img)

            img_qt = QImage(display_img.data, display_img.shape[1], display_img.shape[0], display_img.strides[0], QImage.Format_BGR888)
            self.labels[i].setPixmap(QPixmap.fromImage(img_qt).scaled(640, 360))

            if people_boxes:
                all_bev_points.append(project_to_bev(people_boxes, mtx, rotation_vector, translation_vector))
            else:
                all_bev_points.append(None)

        print(all_bev_points)
        if len(all_bev_points) ==2 and all_bev_points[0] is not None and all_bev_points[1] is not None:
            fused_gd = (np.array(all_bev_points[0])+np.array(all_bev_points[1]))/2
        elif len([v for v in all_bev_points if v is not None]) != 0:
            fused_gd = [v for v in all_bev_points if v is not None][0]
        else:
            fused_gd = None
        
        now = time.time()
        if fused_gd is not None:#换成valid point
            measurement = np.array([[np.float32(fused_gd[0])], [np.float32(fused_gd[1])]])
            kalman.correct(measurement)
            # x = int(BEV_SIZE // 2 + fused_gd[0].item() * 10)
            # y = int(BEV_SIZE - fused_gd[1].item() *10)
            x = (fused_gd[0].item()/20+self.BEV_SIZE/5)*2
            y = (fused_gd[1].item()/20+self.BEV_SIZE/5)*2
            print(x,y)
            if 0 <= x < self.BEV_SIZE and 0 <= y < self.BEV_SIZE:
                self.trajectories.append(((x, y), now))
                if len(self.trajectories)>1:
                    cv2.line(self.bev_canvas, tuple(map(int,self.trajectories[-2][0])), tuple(map(int,self.trajectories[-1][0])), (0,0,255), 2)

            bev_qt = QImage(self.bev_canvas.data, self.BEV_SIZE, self.BEV_SIZE, self.bev_canvas.strides[0], QImage.Format_RGB888)
            self.bev_label.setPixmap(QPixmap.fromImage(bev_qt).scaled(640, 360))



if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = CameraWidget()
    window.show()
    sys.exit(app.exec_())
