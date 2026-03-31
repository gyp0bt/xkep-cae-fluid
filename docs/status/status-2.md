# status-2: 3次元非定常伝熱解析 (FDM) 実装

[← status-index](status-index.md) | [← README](../../README.md)

## 日付

2026-03-31

## 概要

等間隔直交格子（ボクセル）上での3次元非定常伝熱解析を FDM + ガウスザイデル法で実装した。

## 実装内容

### 新規ファイル

| ファイル | 内容 |
|---------|------|
| `xkep_cae_fluid/heat_transfer/__init__.py` | パッケージ公開API |
| `xkep_cae_fluid/heat_transfer/data.py` | HeatTransferInput, HeatTransferResult, BoundarySpec, BoundaryCondition |
| `xkep_cae_fluid/heat_transfer/solver.py` | HeatTransferFDMProcess（SolverProcess継承） |
| `docs/design/heat-transfer-fdm.md` | 設計仕様書 |
| `tests/test_heat_transfer_fdm.py` | APIテスト4件 + 物理テスト5件 |

### 機能詳細

- **支配方程式**: ρC ∂T/∂t = ∇·(k∇T) + q
- **空間離散化**: 中心差分（面間熱伝導率は調和平均）
- **時間離散化**: 陰的オイラー法（後退オイラー）
- **反復法**: ガウスザイデル法
- **領域**: Lx × Ly × Lz の等間隔直交格子（nx × ny × nz ボクセル）
- **材料分布**: セルごとに k, C, q, T0 を配列で指定可能
- **境界条件**: Dirichlet（温度固定）、Neumann（熱流束指定）、Adiabatic（断熱）
- **定常/非定常**: dt=0 で定常、dt>0 で非定常解析

## テスト結果

- テスト数: 25（既存16 + 新規9）
- 契約違反: 0件
- 全テスト PASSED

### 物理テスト一覧

| テスト | 検証内容 | 結果 |
|--------|---------|------|
| 1D定常Dirichlet両端 | 線形温度分布 | rtol < 1% |
| 1D定常Dirichlet-Neumann | 解析解との比較 | rtol < 2% |
| 断熱+均一発熱 | エネルギー保存 | rtol < 1e-6 |
| 1D定常+発熱 | 放物線温度分布 | rtol < 2% |
| 不均一熱伝導率 | 2層構造の界面温度 | rtol < 5% |

## TODO

- [ ] ソルバーの高速化（NumPy ベクトル化 or Numba JIT）— 現在はPython 3重ループで低速
- [ ] Robin境界条件（対流熱伝達）の追加
- [ ] 2D/3D のより本格的なベンチマーク
- [ ] 可視化 PostProcess の実装
- [ ] Phase 2（メッシュ生成・離散化スキーム）への接続

## 設計上の懸念

- 現在のガウスザイデル法はPure Python の3重ループのため大規模問題では極めて低速。NumPyベクトル化またはNumba JITによる高速化が必須。
- _apply_bc_face は未使用関数として残っている（_bc_coefficients に統合済み）。次回クリーンアップ推奨。
