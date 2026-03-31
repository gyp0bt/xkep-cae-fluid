"""非構造化メッシュ読み込み Process（OpenFOAM polyMesh 互換）.

OpenFOAM の constant/polyMesh/ ディレクトリから
points, faces, owner, neighbour, boundary を読み込み、
MeshData として返す。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import numpy as np

from xkep_cae_fluid.core.base import AbstractProcess, ProcessMeta
from xkep_cae_fluid.core.categories import PreProcess
from xkep_cae_fluid.core.data import MeshData

# ---------------------------------------------------------------------------
# 入出力データ
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PolyMeshInput:
    """polyMesh 読み込みの入力.

    Parameters
    ----------
    mesh_dir : str
        polyMesh ディレクトリのパス（points, faces, owner, neighbour を含む）
    """

    mesh_dir: str


@dataclass(frozen=True)
class PolyMeshResult:
    """polyMesh 読み込みの出力.

    Parameters
    ----------
    mesh : MeshData
        読み込まれたメッシュデータ
    boundary_patches : dict[str, dict]
        境界パッチ情報（パッチ名 → {type, nFaces, startFace}）
    """

    mesh: MeshData
    boundary_patches: dict[str, dict]


# ---------------------------------------------------------------------------
# OpenFOAM ファイルパーサ
# ---------------------------------------------------------------------------


def _skip_header(lines: list[str]) -> int:
    """FoamFile ヘッダをスキップして本体の開始行インデックスを返す."""
    i = 0
    n = len(lines)
    # FoamFile ヘッダブロックをスキップ
    while i < n:
        stripped = lines[i].strip()
        if stripped.startswith("FoamFile"):
            # { ... } ブロックを飛ばす
            while i < n and "}" not in lines[i]:
                i += 1
            i += 1  # } の次の行
            break
        i += 1
    return i


def _read_count(lines: list[str], start: int) -> tuple[int, int]:
    """データ件数を読み取り、開き括弧 '(' の次の行インデックスを返す."""
    i = start
    n = len(lines)
    count = 0
    while i < n:
        stripped = lines[i].strip()
        if stripped and not stripped.startswith("//"):
            try:
                count = int(stripped)
                i += 1
                break
            except ValueError:
                pass
        i += 1
    # '(' を見つける
    while i < n:
        if "(" in lines[i]:
            i += 1
            break
        i += 1
    return count, i


def parse_points(text: str) -> np.ndarray:
    """OpenFOAM points ファイルを解析して (n_points, 3) 配列を返す."""
    lines = text.splitlines()
    start = _skip_header(lines)
    n_points, data_start = _read_count(lines, start)

    points = np.zeros((n_points, 3))
    idx = 0
    for i in range(data_start, len(lines)):
        line = lines[i].strip()
        if line == ")":
            break
        # "(x y z)" 形式
        line = line.strip("()")
        parts = line.split()
        if len(parts) == 3:
            points[idx] = [float(p) for p in parts]
            idx += 1
    return points


def parse_faces(text: str) -> list[list[int]]:
    """OpenFOAM faces ファイルを解析して面のリストを返す."""
    lines = text.splitlines()
    start = _skip_header(lines)
    n_faces, data_start = _read_count(lines, start)

    faces: list[list[int]] = []
    for i in range(data_start, len(lines)):
        line = lines[i].strip()
        if line == ")":
            break
        # "4(0 1 5 4)" 形式
        if "(" in line:
            inner = line[line.index("(") + 1 : line.index(")")]
            node_ids = [int(x) for x in inner.split()]
            faces.append(node_ids)
    return faces


def parse_label_list(text: str) -> np.ndarray:
    """OpenFOAM の owner/neighbour ファイルを解析して整数配列を返す."""
    lines = text.splitlines()
    start = _skip_header(lines)
    n_items, data_start = _read_count(lines, start)

    labels = np.zeros(n_items, dtype=np.int64)
    idx = 0
    for i in range(data_start, len(lines)):
        line = lines[i].strip()
        if line == ")":
            break
        if line:
            labels[idx] = int(line)
            idx += 1
    return labels


def parse_boundary(text: str) -> dict[str, dict]:
    """OpenFOAM boundary ファイルを解析してパッチ情報を返す."""
    lines = text.splitlines()
    start = _skip_header(lines)
    _n_patches, data_start = _read_count(lines, start)

    patches: dict[str, dict] = {}
    i = data_start
    n = len(lines)
    while i < n:
        line = lines[i].strip()
        if line == ")":
            break
        # パッチ名
        if line and not line.startswith("//") and line != "{":
            patch_name = line
            i += 1
            # { を見つける
            while i < n and "{" not in lines[i]:
                i += 1
            i += 1
            # パッチの属性を読む
            patch_data: dict[str, str | int] = {}
            while i < n:
                pline = lines[i].strip().rstrip(";")
                if "}" in pline:
                    i += 1
                    break
                parts = pline.split()
                if len(parts) == 2:
                    key, val = parts
                    try:
                        patch_data[key] = int(val)
                    except ValueError:
                        patch_data[key] = val
                i += 1
            patches[patch_name] = patch_data
            continue
        i += 1
    return patches


# ---------------------------------------------------------------------------
# メッシュ計算ヘルパー
# ---------------------------------------------------------------------------


def _compute_face_geometry(
    points: np.ndarray,
    faces: list[list[int]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """面の面積、法線、中心を計算.

    三角形分割で面積と法線を計算する。
    """
    n_faces = len(faces)
    areas = np.zeros(n_faces)
    normals = np.zeros((n_faces, 3))
    centers = np.zeros((n_faces, 3))

    for f_idx, face_nodes in enumerate(faces):
        pts = points[face_nodes]
        center = pts.mean(axis=0)
        centers[f_idx] = center

        # 三角形分割で面積と法線を計算
        area_vec = np.zeros(3)
        for j in range(1, len(face_nodes) - 1):
            v1 = pts[j] - pts[0]
            v2 = pts[j + 1] - pts[0]
            area_vec += 0.5 * np.cross(v1, v2)

        area = np.linalg.norm(area_vec)
        areas[f_idx] = area
        if area > 0:
            normals[f_idx] = area_vec / area

    return areas, normals, centers


def _compute_cell_geometry(
    points: np.ndarray,
    faces: list[list[int]],
    owner: np.ndarray,
    neighbour: np.ndarray,
    n_cells: int,
) -> tuple[np.ndarray, np.ndarray]:
    """セルの体積と中心を計算.

    面中心をもとにセル中心を近似し、発散定理で体積を計算する。
    """
    # セル中心の初期推定: 所属する面の中心の平均
    cell_centers = np.zeros((n_cells, 3))
    cell_face_count = np.zeros(n_cells)

    for f_idx, face_nodes in enumerate(faces):
        pts = points[face_nodes]
        fc = pts.mean(axis=0)
        o = owner[f_idx]
        cell_centers[o] += fc
        cell_face_count[o] += 1
        if f_idx < len(neighbour):
            nb = neighbour[f_idx]
            cell_centers[nb] += fc
            cell_face_count[nb] += 1

    safe_count = np.maximum(cell_face_count, 1)
    cell_centers /= safe_count[:, np.newaxis]

    # セル体積: 発散定理 V = (1/3) * Σ (r_f · n_f * A_f)
    cell_volumes = np.zeros(n_cells)
    for f_idx, face_nodes in enumerate(faces):
        pts = points[face_nodes]
        fc = pts.mean(axis=0)

        # 面面積ベクトル
        area_vec = np.zeros(3)
        for j in range(1, len(face_nodes) - 1):
            v1 = pts[j] - pts[0]
            v2 = pts[j + 1] - pts[0]
            area_vec += 0.5 * np.cross(v1, v2)

        vol_contrib = np.dot(fc, area_vec) / 3.0
        cell_volumes[owner[f_idx]] += vol_contrib
        if f_idx < len(neighbour):
            cell_volumes[neighbour[f_idx]] -= vol_contrib

    cell_volumes = np.abs(cell_volumes)
    return cell_volumes, cell_centers


# ---------------------------------------------------------------------------
# PolyMeshReaderProcess
# ---------------------------------------------------------------------------


class PolyMeshReaderProcess(PreProcess["PolyMeshInput", "PolyMeshResult"]):
    """OpenFOAM polyMesh 読み込み Process.

    constant/polyMesh/ ディレクトリから非構造化メッシュを読み込み、
    MeshData として返す。
    """

    meta: ClassVar[ProcessMeta] = ProcessMeta(
        name="PolyMeshReader",
        module="pre",
        version="0.1.0",
        document_path="../../docs/design/polymesh-reader.md",
        stability="experimental",
    )
    uses: ClassVar[list[type[AbstractProcess]]] = []

    def process(self, input_data: PolyMeshInput) -> PolyMeshResult:
        """polyMesh ディレクトリからメッシュを読み込む."""
        mesh_dir = Path(input_data.mesh_dir)

        # ファイル読み込み
        points = parse_points((mesh_dir / "points").read_text())
        faces_list = parse_faces((mesh_dir / "faces").read_text())
        owner = parse_label_list((mesh_dir / "owner").read_text())
        neighbour = parse_label_list((mesh_dir / "neighbour").read_text())
        boundary_patches = parse_boundary((mesh_dir / "boundary").read_text())

        n_cells = int(owner.max()) + 1
        n_internal_faces = len(neighbour)

        # 面の幾何情報
        face_areas, face_normals, face_centers = _compute_face_geometry(points, faces_list)

        # セルの幾何情報
        cell_volumes, cell_centers = _compute_cell_geometry(
            points, faces_list, owner, neighbour, n_cells
        )

        # connectivity: セルごとのノードを集める（ユニーク）
        cell_nodes: list[set[int]] = [set() for _ in range(n_cells)]
        for f_idx, face_nodes in enumerate(faces_list):
            cell_nodes[owner[f_idx]].update(face_nodes)
            if f_idx < n_internal_faces:
                cell_nodes[neighbour[f_idx]].update(face_nodes)

        max_cell_nodes = max((len(cn) for cn in cell_nodes), default=0)
        connectivity = np.full((n_cells, max_cell_nodes), -1, dtype=np.int64)
        for c, nodes in enumerate(cell_nodes):
            sorted_nodes = sorted(nodes)
            connectivity[c, : len(sorted_nodes)] = sorted_nodes

        mesh = MeshData(
            node_coords=points,
            connectivity=connectivity,
            cell_volumes=cell_volumes,
            face_areas=face_areas,
            face_normals=face_normals,
            face_centers=face_centers,
            cell_centers=cell_centers,
            face_owner=owner,
            face_neighbour=neighbour,
        )

        return PolyMeshResult(mesh=mesh, boundary_patches=boundary_patches)
