# status-8: status-7 TODO 全消化 — 離散化スキーム + MeshData対応 + polyMesh読込

[← status-index](status-index.md) | [← README](../../README.md)

## 日付

2026-03-31

## 概要

status-7 の TODO 6件を全て消化。
中心差分拡散スキーム、1次風上対流スキーム、伝熱ソルバーの MeshData 対応リファクタリング、
非構造化メッシュ読み込み Process、Numba JIT 性能ベンチマーク、CI オプション依存テストを実装。

## 実装内容

### 変更ファイル

| ファイル | 変更内容 |
|---------|---------|
| `.github/workflows/ci.yml` | `test-optional-deps` ジョブ追加（pyamg + numba テスト） |
| `examples/benchmark_solver_methods.py` | ソルバー手法別ベンチマークスクリプト新規作成 |
| `xkep_cae_fluid/core/strategies/diffusion.py` | CentralDiffusionScheme 新規作成 |
| `xkep_cae_fluid/core/strategies/convection.py` | UpwindConvectionScheme 新規作成 |
| `xkep_cae_fluid/core/strategies/__init__.py` | 具象スキームのエクスポート追加 |
| `xkep_cae_fluid/heat_transfer/data.py` | dx_array/dy_array/dz_array + from_mesh() 追加 |
| `xkep_cae_fluid/heat_transfer/solver_sparse.py` | build_sparse_system_nonuniform() + _build_system() 追加 |
| `xkep_cae_fluid/core/mesh_reader.py` | PolyMeshReaderProcess 新規作成 |
| `xkep_cae_fluid/core/__init__.py` | PolyMeshReader エクスポート追加 |
| `tests/test_diffusion_scheme.py` | 拡散スキームテスト10件 新規作成 |
| `tests/test_convection_scheme.py` | 対流スキームテスト9件 新規作成 |
| `tests/test_heat_transfer_fdm.py` | MeshData対応テスト3件 追加 |
| `tests/test_polymesh_reader.py` | polyMesh読込テスト14件 新規作成 |
| `docs/design/polymesh-reader.md` | PolyMeshReaderProcess 設計文書 新規作成 |
| `docs/roadmap.md` | Phase 2 タスク更新 |
| `README.md` | テスト数・プロセス数・パッケージ構成更新 |

### 機能詳細

#### 1. CI オプション依存テスト

- `test-optional-deps` ジョブ: Python 3.11 で pyamg + numba をインストールし AMG/Numba テストを実行
- 既存の `test` ジョブ（Python 3.10-3.12）はそのまま維持

#### 2. Numba JIT 性能ベンチマーク

- `examples/benchmark_solver_methods.py`: Python GS / Vectorized Jacobi / Numba GS の比較
- 結果: Numba GS は Python GS 比 **176〜256倍** の高速化
- YAML 出力で再現性を確保

#### 3. 中心差分拡散スキーム（CentralDiffusionScheme）

- DiffusionSchemeStrategy Protocol の具象実装
- MeshData の面情報（face_owner, face_neighbour, face_areas, cell_centers）を使用
- 面間拡散係数は調和平均、セル中心間距離ベースの離散化
- flux() と matrix_coefficients() の両メソッドを提供
- 対称行列（SPD）、保存性（内部セルの行和ゼロ）を検証済み

#### 4. 1次風上対流スキーム（UpwindConvectionScheme）

- ConvectionSchemeStrategy Protocol の具象実装
- 面質量流束 ṁ_f = (u_f · n_f) * A_f に基づく風上値選択
- M-matrix 性（対角優位、非対角非正）を保証
- flux() と matrix_coefficients() の両メソッドを提供

#### 5. 伝熱ソルバー MeshData 対応リファクタリング

- HeatTransferInput に `dx_array`, `dy_array`, `dz_array` オプションフィールド追加
- `from_mesh()` クラスメソッドで StructuredMeshResult から入力を自動生成
- `build_sparse_system_nonuniform()`: 不等間隔格子用の疎行列組立
- `_build_system()`: 等間隔/不等間隔を自動判定してディスパッチ
- 等間隔格子の既存コードは完全に互換維持

#### 6. 非構造化メッシュ読み込み Process（PolyMeshReaderProcess）

- OpenFOAM `constant/polyMesh/` の ASCII 形式を読み込み
- 対応ファイル: points, faces, owner, neighbour, boundary
- 面の幾何情報は三角形分割、セル体積は発散定理で計算
- MeshData + boundary_patches として出力

## テスト結果

- テスト数: **124**（既存88 + 拡散10 + 対流9 + MeshData 3 + polyMesh 14）
- 契約違反: **0件**（6プロセス登録）
- 全テスト PASSED

## status-7 TODO 消化状況

- [x] 既存伝熱ソルバーの MeshData 対応リファクタリング
- [x] 非構造化メッシュ読み込み Process（OpenFOAM polyMesh 互換）
- [x] 中心差分拡散スキーム実装（MeshData ベース）
- [x] 1次風上対流スキーム実装
- [x] Numba JIT の性能ベンチマーク
- [x] CI に pyamg/numba のオプション依存テストを追加

## TODO

- [ ] TVD 対流スキーム実装（van Leer, Superbee）
- [ ] 非直交補正付き拡散スキーム実装
- [ ] PolyMeshReader のバイナリ形式対応
- [ ] Phase 3: SIMPLE ソルバー着手（運動量方程式アセンブリ）
- [ ] PyAMG の AMG 構築キャッシュ化（非定常解析の高速化）

## 設計上の懸念

- CentralDiffusionScheme と UpwindConvectionScheme は内部面のみ対応。境界面の処理は Phase 3 でソルバーに統合する際に追加が必要
- HeatTransferInput.from_mesh() の不等間隔対応は solver_sparse（direct/bicgstab/amg）のみ。反復法（jacobi/numba）は依然として等間隔前提
- PolyMeshReaderProcess はセル体積の計算精度が発散定理近似のため、非凸セルでは誤差が大きくなる可能性

## 開発運用メモ

- 効果的: Strategy Protocol が事前定義済みだったため、具象スキーム実装が Protocol 準拠で迷いなく進んだ
- 効果的: StructuredMeshProcess のメッシュ生成テストデータが既にあったため、スキームのテストが容易に作成できた
- 効果的: ステップバイステップ（スキーム → テスト → lint → コミット）で各機能を独立にコミットし、問題の切り分けが容易
- 注意: OpenFOAM polyMesh のテストデータ作成は手動で面番号を管理する必要があり、ミスしやすい。自動生成ツールの整備を検討
