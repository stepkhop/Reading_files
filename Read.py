import os
import numpy as np
import trimesh
from scipy.spatial import KDTree
from multiprocessing import Pool


# Функция для чтения файла сетки (nodes.txt)
def read_mesh_file(filename):
    """
    Читает файл с узлами и элементами.
    Формат: первая строка - число узлов и элементов.
    Затем строки узлов: id x y z
    Пропуск строки.
    Затем строки элементов: ... element_id node_ids...
    """
    with open(filename, 'r') as file:
        line = file.readline().strip().split()
        num_nodes, num_elements = map(int, line)

        nodes = []
        for _ in range(num_nodes):
            line = file.readline().strip().split()
            node_id = int(line[0])
            x, y, z = map(float, line[1:4])
            nodes.append((node_id, x, y, z))

        line = file.readline()

        elements = []
        for _ in range(num_elements):
            line = file.readline().strip().split()
            element_id = int(line[10])
            node_ids = list(dict.fromkeys(map(int, line[11:19])))
            elements.append((element_id, node_ids))

    return num_nodes, num_elements, nodes, elements


# Функция для чтения вершин из .ply файлов в папке
def read_ply_vertices(folder_path):
    """
    Читает все .ply файлы в указанной папке и возвращает список массивов вершин.
    Каждый элемент списка - вершины одного кластера (облака точек).
    """
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Папка {folder_path} не существует.")

    ply_files = [f for f in os.listdir(folder_path) if f.endswith('.ply')]
    all_vertices = []
    for ply_file in ply_files:
        file_path = os.path.join(folder_path, ply_file)
        mesh = trimesh.load(file_path)
        vertices = np.array(mesh.vertices)
        all_vertices.append(vertices)

    return all_vertices


# Функция для нахождения центра конечного элемента (КЭ с 4 вершинами)
def find_element_center(element, nodes):
    """
    Вычисляет центр КЭ как среднее координат его 4 вершин.
    element: (element_id, [node_ids])
    nodes: список узлов (id, x, y, z)
    """
    node_ids = element[1]
    if len(node_ids) != 4:
        raise ValueError("Элемент должен иметь ровно 4 вершины.")

    node_coords = [np.array(nodes[i - 1][1:4]) for i in node_ids]
    center = np.mean(node_coords, axis=0)
    return center


# Функция для расчета расстояния между двумя точками
def distance_between_points(point1, point2):
    """
    Евклидово расстояние между двумя точками (x1,y1,z1) и (x2,y2,z2).
    """
    return np.linalg.norm(np.array(point1) - np.array(point2))


# Функция обработки одного элемента для параллельного выполнения
def process_element(args):
    element, nodes, kd_trees = args
    element_id = element[0]
    try:
        center = find_element_center(element, nodes)
    except ValueError as e:
        print(f"Ошибка для элемента {element_id}: {e}")
        return None

    min_dist = float('inf')
    closest_cluster = -1
    for cluster_idx, tree in enumerate(kd_trees):
        if tree is None:
            continue
        dist, _ = tree.query(center, k=1)  # Поиск только ближайшей точки
        if dist < min_dist:
            min_dist = dist
            closest_cluster = cluster_idx + 1

    return (element_id, closest_cluster) if closest_cluster != -1 else None


# Основная логика
def main():
    # Пути к файлам (все в одной папке, Partitions - поддиректория)
    filename = "nodes.txt"
    folder_path = "Partitions"

    # Чтение сетки
    try:
        num_nodes, num_elements, nodes, elements = read_mesh_file(filename)
        print(f"Число узлов: {num_nodes}")
        print(f"Число элементов: {num_elements}")
    except FileNotFoundError as e:
        print(f"Ошибка: {e}")
        return

    # Чтение вершин из ply (облака точек, кластеры)
    try:
        all_vertices = read_ply_vertices(folder_path)
        num_clusters = len(all_vertices)
        print(f"Найдено {num_clusters} кластеров (ply файлов).")
        if num_clusters == 0:
            print("Нет ply файлов в папке Partitions.")
            return
    except FileNotFoundError as e:
        print(f"Ошибка: {e}")
        return

    # Построение KDTree один раз для каждого кластера
    kd_trees = [KDTree(vertices) if len(vertices) > 0 else None for vertices in all_vertices]

    # Параллельная обработка элементов
    with Pool() as pool:
        results = pool.map(process_element, [(elem, nodes, kd_trees) for elem in elements])

    # Фильтрация и вывод результатов
    results = [r for r in results if r is not None]
    print("Список: Номер КЭ - номер кластера")
    for elem_id, cluster_id in results:
        print(f"{elem_id} - {cluster_id}")


# Запуск основной функции
if __name__ == "__main__":
    main()