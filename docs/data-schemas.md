# データスキーマ仕様

[<- README](../README.md) | [<- docs](README.md)

## 概要

xkep-cae-fluid のプロセス間データ受け渡しに使用する `frozen dataclass` の仕様。
全データ型は `xkep_cae_fluid.core.data` に定義される。

## 一覧

| データ型 | カテゴリ | 用途 |
|---------|---------|------|
| MeshData | メッシュ | 構造化/非構造化メッシュの節点・セル・面情報 |
| BoundaryData | 境界条件 | パッチ型境界条件（Dirichlet/Neumann/壁等） |
| FluidProperties | 物性値 | 密度・粘度・熱伝導率等 |
| FlowFieldData | 流れ場 | 速度・圧力・温度・乱流量 |
| SolverInputData | ソルバー入力 | メッシュ + 境界条件 + 物性 + ソルバー設定 |
| SolverResultData | ソルバー出力 | 収束後の流れ場 + 診断情報 |
| VerifyInput | 検証入力 | ソルバー結果 + 期待値 |
| VerifyResult | 検証出力 | 合否 + 比較結果 |

## MeshData

FDM（構造化格子）と FVM（非構造化格子）の両方に対応する。

### 構造化格子の場合

```python
mesh = MeshData(
    node_coords=coords,          # (nx*ny*nz, 3)
    connectivity=conn,           # (n_cells, 8) for hexahedra
    cell_volumes=volumes,        # (n_cells,)
    dimensions=(nx, ny, nz),     # 構造化を示すフラグ
)
assert mesh.is_structured == True
```

### 非構造化格子（FVM）の場合

```python
mesh = MeshData(
    node_coords=coords,
    connectivity=conn,
    cell_volumes=volumes,
    face_areas=areas,            # (n_faces,)
    face_normals=normals,        # (n_faces, 3)
    face_centers=centers,        # (n_faces, 3)
    cell_centers=cell_centers,   # (n_cells, 3)
    face_owner=owner,            # (n_faces,) 各面のオーナーセル
    face_neighbour=neighbour,    # (n_internal_faces,) 内部面の隣接セル
)
assert mesh.is_structured == False
```

### プロパティ

| プロパティ | 型 | 説明 |
|-----------|-----|------|
| `n_nodes` | int | 節点数 |
| `n_cells` | int | セル数 |
| `ndim` | int | 空間次元（2 or 3） |
| `is_structured` | bool | 構造化格子か |

## BoundaryData

FVM のパッチ型境界条件を表現する。

```python
boundary = BoundaryData(
    patch_faces={
        "inlet": np.array([0, 1, 2]),
        "outlet": np.array([10, 11, 12]),
        "wall": np.array([3, 4, 5, 6, 7, 8, 9]),
    },
    patch_types={
        "inlet": "dirichlet",
        "outlet": "neumann",
        "wall": "wall",
    },
    patch_values={
        "inlet": np.array([1.0, 0.0, 0.0]),  # velocity
        "outlet": 0.0,                         # pressure gradient
        "wall": np.array([0.0, 0.0, 0.0]),    # no-slip
    },
)
```

### 境界条件種別

| 種別 | 説明 |
|------|------|
| `dirichlet` | 値固定（速度入口、温度固定壁等） |
| `neumann` | 勾配固定（自然流出等） |
| `wall` | 壁面（no-slip, 壁関数適用） |
| `symmetry` | 対称面 |
| `inlet` | 流入境界（速度指定） |
| `outlet` | 流出境界（圧力指定） |

## FluidProperties

```python
fluid = FluidProperties(
    density=1.225,           # kg/m^3 (空気)
    viscosity=1.789e-5,      # Pa*s
    specific_heat=1006.0,    # J/(kg*K)
    thermal_conductivity=0.0242,  # W/(m*K)
)
print(fluid.kinematic_viscosity)  # nu = mu / rho
```

### 非ニュートン流体

```python
fluid = FluidProperties(
    density=1000.0,
    viscosity=0.001,
    power_law_n=0.5,         # せん断減粘性（n < 1）
    power_law_k=0.1,
)
```

## SolverInputData

ソルバーへの統一入力。定常/非定常は `dt` で自動判定。

```python
config = SolverInputData(
    mesh=mesh,
    boundary=boundary,
    fluid=fluid,
    dt=0.0,                      # 0.0 = 定常
    coupling_method="SIMPLE",
    relax_velocity=0.7,
    relax_pressure=0.3,
    max_iterations=1000,
    tol_residual=1e-6,
)
assert config.is_transient == False
```

## SolverResultData

```python
@dataclass(frozen=True)
class SolverResultData:
    field: FlowFieldData         # 最終流れ場
    converged: bool
    n_iterations: int
    residual_history: tuple      # 残差履歴
    elapsed_seconds: float
    n_timesteps: int             # 非定常の場合
    time_history: tuple          # 時刻履歴
    field_history: tuple         # スナップショット
```
