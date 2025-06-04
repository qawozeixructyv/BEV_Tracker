import cv2

def draw_circle(image, center, radius=10, color=(0, 255, 0), thickness=-1):
    """在图像上画圆点（原点）"""
    # 使用 cv2.circle 绘制圆点
    cv2.circle(image, center, radius, color, thickness)

# 画网格的函数
def draw_grid(image, grid_size=50, color=(200, 200, 200)):
    """在图像上画网格"""
    height, width = image.shape[:2]
    
    # 横向网格线
    for y in range(0, height, grid_size):
        cv2.line(image, (0, y), (width, y), color, 1)
    
    # 纵向网格线
    for x in range(0, width, grid_size):
        cv2.line(image, (x, 0), (x, height), color, 1)