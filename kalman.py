import numpy as np
from collections import deque

class EnhancedKalmanFilter:
    def __init__(self, dt=0.3, max_speed=100.0, history_size=5):
        """
        增强版卡尔曼滤波器，添加速度约束
        :param dt: 时间步长（秒）
        :param max_speed: 最大允许速度（mm/秒）
        :param history_size: 历史轨迹记录长度
        """
        # 初始化标准卡尔曼滤波器参数
        self.state = np.zeros((4, 1))  # [x, y, vx, vy]都初始化为0
        self.F = np.array([[1,0,dt,0], [0,1,0,dt], [0,0,1,0], [0,0,0,1]])
        self.H = np.array([[1,0,0,0], [0,1,0,0]])
        self.Q = np.eye(4) * 0.01
        self.R = np.eye(2) * 0.1
        self.P = np.eye(4)
        
        # 增强参数
        self.max_speed = max_speed
        self.dt = dt
        self.trajectory = deque(maxlen=history_size)  # 轨迹历史记录

    def _apply_speed_constraint(self):
        """应用速度约束"""
        # 获取当前速度
        vx, vy = self.state[2][0], self.state[3][0]
        speed = np.sqrt(vx**2 + vy**2)
        
        # 速度限制
        if speed > self.max_speed:
            scale = self.max_speed / speed
            self.state[2] *= scale
            self.state[3] *= scale

    def _smooth_with_history(self):
        """基于历史轨迹的平滑"""
        if len(self.trajectory) >= 2:
            # 计算平均速度
            avg_vx = np.mean([p[2] for p in self.trajectory])
            avg_vy = np.mean([p[3] for p in self.trajectory])
            
            # 速度加权调整
            self.state[2] = 0.7 * self.state[2] + 0.3 * avg_vx
            self.state[3] = 0.7 * self.state[3] + 0.3 * avg_vy


    def correct(self, measurement):
        """带轨迹平滑的更新"""
        measurement = np.array(measurement).reshape((2, 1))
        self.state = np.dot(self.F, self.state)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        self._apply_speed_constraint()
        # 卡尔曼增益计算
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        
        # 状态更新
        self.state += np.dot(K, (measurement - np.dot(self.H, self.state)))
        self.P = np.dot((np.eye(4) - np.dot(K, self.H)), self.P)
        
        # 记录历史状态
        self.trajectory.append(self.state.copy())
        
        # 轨迹平滑
        if len(self.trajectory) >= 2:
            self._smooth_with_history()
        
        return self.state[:2].flatten()

def filter_abnormal_points(valid_points, points, prev_point, speed_threshold=1000.0, dt=0.3):
    """
    带速度约束的异常点过滤
    :param points: 时序点列表（需包含时间戳）
    :param speed_threshold: 最大允许速度（米/秒）
    :param dt: 时间间隔（秒）
    """
    
    
    for i, current_point in enumerate(points):
        if current_point is None:
            continue
            
        # 首次出现点
        if prev_point[i] is None:
            valid_points.append(current_point)
            prev_point[i] = current_point
            continue
            
        # 计算瞬时速度
        dx = current_point[0] - prev_point[i][0]
        dy = current_point[1] - prev_point[i][1]
        distance = np.sqrt(dx**2 + dy**2)
        speed = distance / dt
        print(f"Frame {i}: Speed = {speed} mm/s")
        
        # 速度验证
        if speed <= speed_threshold:
            valid_points.append(current_point)
            prev_point[i] = current_point
        else:
            # 使用线性插值替代异常点
            if len(valid_points) >= 1:
                avg_step = np.median(np.diff(valid_points[-10:], axis=0), axis=0)
                
                interpolated = valid_points[-1] + avg_step
                print(valid_points[-3:])
    
                print("Previous Point:", prev_point[i])
                print(valid_points)
                print("Interpolated Point:", interpolated)
                valid_points.append(tuple(interpolated))
                prev_point[i] = tuple(interpolated)                # prev_point[i] = current_point
    
    return valid_points,prev_point