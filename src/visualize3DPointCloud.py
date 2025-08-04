import scenepic as sp
import numpy as np


def create_coordinate_axes(size=1.0):
    """
    手动创建坐标轴网格
    """
    mesh = sp.Mesh()

    # 定义坐标轴的顶点和线条
    vertices = []
    triangles = []
    colors = []

    # X轴 (红色)
    x_start = np.array([0, 0, 0], dtype=np.float32)
    x_end = np.array([size, 0, 0], dtype=np.float32)

    # Y轴 (绿色)
    y_start = np.array([0, 0, 0], dtype=np.float32)
    y_end = np.array([0, size, 0], dtype=np.float32)

    # Z轴 (蓝色)
    z_start = np.array([0, 0, 0], dtype=np.float32)
    z_end = np.array([0, 0, size], dtype=np.float32)

    # 创建轴线的粗细
    thickness = size * 0.02

    # X轴线条
    x_vertices, x_triangles, x_colors = create_line_mesh(x_start, x_end, thickness, [1.0, 0.0, 0.0])
    add_mesh_data(vertices, triangles, colors, x_vertices, x_triangles, x_colors)

    # Y轴线条
    y_vertices, y_triangles, y_colors = create_line_mesh(y_start, y_end, thickness, [0.0, 1.0, 0.0])
    add_mesh_data(vertices, triangles, colors, y_vertices, y_triangles, y_colors)

    # Z轴线条
    z_vertices, z_triangles, z_colors = create_line_mesh(z_start, z_end, thickness, [0.0, 0.0, 1.0])
    add_mesh_data(vertices, triangles, colors, z_vertices, z_triangles, z_colors)

    if len(vertices) > 0:
        vertices = np.array(vertices, dtype=np.float32)
        triangles = np.array(triangles, dtype=np.uint32)
        colors = np.array(colors, dtype=np.float32)

        mesh.add_mesh_without_normals(vertices, triangles, colors)

    return mesh


def create_line_mesh(start, end, thickness, color):
    """
    创建线条的网格表示
    """
    # 计算线条方向
    direction = end - start
    length = np.linalg.norm(direction)

    if length == 0:
        return [], [], []

    direction = direction / length

    # 创建垂直向量
    if abs(direction[2]) < 0.9:
        perpendicular = np.cross(direction, [0, 0, 1])
    else:
        perpendicular = np.cross(direction, [1, 0, 0])

    perpendicular = perpendicular / np.linalg.norm(perpendicular)

    # 创建第二个垂直向量
    perpendicular2 = np.cross(direction, perpendicular)
    perpendicular2 = perpendicular2 / np.linalg.norm(perpendicular2)

    # 创建圆柱体的顶点
    vertices = []
    triangles = []
    colors = []

    segments = 8  # 圆柱体的分段数

    for i in range(segments):
        angle = 2 * np.pi * i / segments
        offset = thickness * (np.cos(angle) * perpendicular + np.sin(angle) * perpendicular2)

        # 起点圆周上的点
        v1 = start + offset
        # 终点圆周上的点
        v2 = end + offset

        vertices.extend([v1, v2])
        colors.extend([color, color])

    # 创建圆柱体表面的三角形
    for i in range(segments):
        next_i = (i + 1) % segments

        # 当前段的四个顶点索引
        v1 = i * 2  # 起点圆周
        v2 = i * 2 + 1  # 终点圆周
        v3 = next_i * 2  # 下一个起点圆周
        v4 = next_i * 2 + 1  # 下一个终点圆周

        # 两个三角形组成矩形面
        triangles.extend([
            [v1, v2, v3],
            [v2, v4, v3]
        ])

    return vertices, triangles, colors


def add_mesh_data(vertices_list, triangles_list, colors_list, new_vertices, new_triangles, new_colors):
    """
    将新的网格数据添加到现有列表中
    """
    if len(new_vertices) == 0:
        return

    base_idx = len(vertices_list)
    vertices_list.extend(new_vertices)

    # 调整新三角形的顶点索引
    for triangle in new_triangles:
        triangles_list.append([idx + base_idx for idx in triangle])

    colors_list.extend(new_colors)


def visualize_3d_scene(points, colors=None, camera_positions=None):
    """
    使用 Scenepic 可视化 3D 点云和相机位置
    
    参数:
    points: numpy array of shape (N, 3) - 3D 点坐标
    colors: numpy array of shape (N, 3) - RGB 颜色值 (可选)
    camera_positions: numpy array of shape (M, 3) - 相机位置 (可选)
    """
    # Create Scene
    scene = sp.Scene()

    # Create 3D Mapping canvas
    canvas3d = scene.create_canvas_3d(width=800, height=600)

    # Frame
    frame = canvas3d.create_frame()

    # Checked flag #1
    if colors is None:
        colors = np.ones_like(points) * [0.7, 0.7, 0.7]  # 灰色

    # Create point cloud mesh - 为每个点创建小三角形来可视化
    mesh = sp.Mesh()

    # 创建点云的三角形 - 每个点用两个三角形组成的小方形表示
    vertices = []
    triangles = []
    vertex_colors = []

    for i, point in enumerate(points):
        # 为每个点创建一个小的正方形 (两个三角形)
        size = 0.05  # 点的大小

        # 四个顶点组成正方形
        v1 = point + np.array([-size, -size, 0])
        v2 = point + np.array([size, -size, 0])
        v3 = point + np.array([size, size, 0])
        v4 = point + np.array([-size, size, 0])

        base_idx = len(vertices)
        vertices.extend([v1, v2, v3, v4])

        # 两个三角形
        triangles.extend([
            [base_idx, base_idx + 1, base_idx + 2],
            [base_idx, base_idx + 2, base_idx + 3]
        ])

        # 每个顶点使用相同的颜色
        color = colors[i].astype(np.float32)
        vertex_colors.extend([color, color, color, color])

    vertices = np.array(vertices, dtype=np.float32)
    triangles = np.array(triangles, dtype=np.uint32)
    vertex_colors = np.array(vertex_colors, dtype=np.float32)

    # 添加网格到场景
    mesh.add_mesh_without_normals(vertices, triangles, vertex_colors)
    frame.add_mesh(mesh)

    # 添加相机位置
    if camera_positions is not None:
        camera_mesh = sp.Mesh()
        cam_vertices = []
        cam_triangles = []
        cam_colors = []

        for i, cam_pos in enumerate(camera_positions):
            # 为每个相机位置创建一个较大的正方形
            size = 0.2  # 相机标记的大小

            v1 = cam_pos + np.array([-size, -size, 0])
            v2 = cam_pos + np.array([size, -size, 0])
            v3 = cam_pos + np.array([size, size, 0])
            v4 = cam_pos + np.array([-size, size, 0])

            base_idx = len(cam_vertices)
            cam_vertices.extend([v1, v2, v3, v4])

            # 两个三角形
            cam_triangles.extend([
                [base_idx, base_idx + 1, base_idx + 2],
                [base_idx, base_idx + 2, base_idx + 3]
            ])

            # 红色
            red_color = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            cam_colors.extend([red_color, red_color, red_color, red_color])

        cam_vertices = np.array(cam_vertices, dtype=np.float32)
        cam_triangles = np.array(cam_triangles, dtype=np.uint32)
        cam_colors = np.array(cam_colors, dtype=np.float32)

        camera_mesh.add_mesh_without_normals(cam_vertices, cam_triangles, cam_colors)
        frame.add_mesh(camera_mesh)

    # 添加坐标轴
    # frame.add_coordinate_axes(size=1.0)
    coordinate_axes_mesh = create_coordinate_axes(size=2.0)
    frame.add_mesh(coordinate_axes_mesh)

    # Save
    scene.save_as_html("visualization.html")
    print("3D可视化已保存到 visualization.html")


def visualization():
    # 创建一个螺旋形的点云
    t = np.linspace(0, 10, 1000)
    x = t * np.cos(t)
    y = t * np.sin(t)
    z = t
    points = np.column_stack((x, y, z))

    # 创建渐变颜色
    colors = np.zeros((len(points), 3))
    colors[:, 0] = np.linspace(0, 1, len(points))  # R
    colors[:, 1] = np.linspace(1, 0, len(points))  # G
    colors[:, 2] = np.linspace(0.5, 0.5, len(points))  # B

    # 创建一些示例相机位置
    camera_positions = np.array([
        [0, 0, 0],
        [5, 5, 5],
        [10, 10, 10],
        [-5, -5, -5]
    ])

    # 可视化
    visualize_3d_scene(points, colors, camera_positions)


if __name__ == "__main__":
    visualization()
