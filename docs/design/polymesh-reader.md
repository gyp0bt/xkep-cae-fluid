# PolyMeshReaderProcess 設計文書

[← README](../../README.md)

## 概要

OpenFOAM の `constant/polyMesh/` ディレクトリ形式の非構造化メッシュを
読み込み、MeshData として返す PreProcess。

## 入出力

### PolyMeshInput
- `mesh_dir`: polyMesh ディレクトリのパス

### PolyMeshResult
- `mesh`: MeshData（ノード座標、セル接続、体積、面情報）
- `boundary_patches`: 境界パッチ情報（パッチ名 → type, nFaces, startFace）

## 対応ファイル

| ファイル | 内容 |
|---------|------|
| `points` | ノード座標 (n_points, 3) |
| `faces` | 面のノードリスト |
| `owner` | 各面の owner セル |
| `neighbour` | 内部面の neighbour セル |
| `boundary` | 境界パッチ定義 |

## アルゴリズム

1. 各ファイルを OpenFOAM ASCII 形式でパース
2. 面の幾何情報（面積、法線、中心）を三角形分割で計算
3. セルの幾何情報（体積、中心）を発散定理で計算
4. connectivity をセルごとのユニークノードで構築

## 制限事項

- ASCII 形式のみ対応（バイナリ形式は非対応）
- 圧縮形式は非対応
