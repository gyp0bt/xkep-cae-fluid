"""中心差分拡散スキームのテスト.

CentralDiffusionScheme が DiffusionSchemeStrategy Protocol を満たし、
MeshData ベースの FVM 離散化が正しく動作することを検証する。
"""

from __future__ import annotations

import numpy as np

from xkep_cae_fluid.core.mesh import StructuredMeshInput, StructuredMeshProcess
from xkep_cae_fluid.core.strategies.diffusion import CentralDiffusionScheme
from xkep_cae_fluid.core.strategies.protocols import DiffusionSchemeStrategy


class TestCentralDiffusionAPI:
    """API テスト."""

    def test_satisfies_protocol(self):
        """Protocol を満たしているか."""
        scheme = CentralDiffusionScheme()
        assert isinstance(scheme, DiffusionSchemeStrategy)

    def test_matrix_shape(self):
        """行列のサイズが n_cells x n_cells."""
        mesh_result = StructuredMeshProcess().process(
            StructuredMeshInput(Lx=1.0, Ly=1.0, Lz=1.0, nx=3, ny=3, nz=3)
        )
        scheme = CentralDiffusionScheme()
        A = scheme.matrix_coefficients(1.0, mesh_result.mesh)
        n = mesh_result.mesh.n_cells
        assert A.shape == (n, n)

    def test_matrix_symmetric(self):
        """拡散行列は対称."""
        mesh_result = StructuredMeshProcess().process(
            StructuredMeshInput(Lx=1.0, Ly=1.0, Lz=1.0, nx=4, ny=4, nz=4)
        )
        scheme = CentralDiffusionScheme()
        A = scheme.matrix_coefficients(1.0, mesh_result.mesh)
        diff = A - A.T
        assert diff.nnz == 0 or np.max(np.abs(diff.data)) < 1e-14

    def test_flux_shape(self):
        """フラックスのサイズが n_cells."""
        mesh_result = StructuredMeshProcess().process(
            StructuredMeshInput(Lx=1.0, Ly=1.0, Lz=1.0, nx=3, ny=3, nz=3)
        )
        scheme = CentralDiffusionScheme()
        phi = np.ones(mesh_result.mesh.n_cells)
        f = scheme.flux(phi, 1.0, mesh_result.mesh)
        assert f.shape == (mesh_result.mesh.n_cells,)


class TestCentralDiffusionPhysics:
    """物理テスト."""

    def test_uniform_field_zero_flux(self):
        """一様場の拡散フラックスはゼロ."""
        mesh_result = StructuredMeshProcess().process(
            StructuredMeshInput(Lx=1.0, Ly=1.0, Lz=1.0, nx=5, ny=5, nz=5)
        )
        scheme = CentralDiffusionScheme()
        phi = np.ones(mesh_result.mesh.n_cells) * 100.0
        f = scheme.flux(phi, 1.0, mesh_result.mesh)
        np.testing.assert_allclose(f, 0.0, atol=1e-14)

    def test_linear_field_1d(self):
        """1D線形場の内部セルでは拡散フラックスがゼロ（2次精度）."""
        nx = 10
        mesh_result = StructuredMeshProcess().process(
            StructuredMeshInput(Lx=1.0, Ly=0.1, Lz=0.1, nx=nx, ny=1, nz=1)
        )
        mesh = mesh_result.mesh
        # x方向に線形温度分布
        phi = mesh.cell_centers[:, 0]  # T = x

        scheme = CentralDiffusionScheme()
        f = scheme.flux(phi, 1.0, mesh)

        # 内部セル（境界面のない中間セル）では拡散フラックス = 0
        # 端のセルは境界面がないため、内部面のみで計算
        # 等間隔1D格子の内部セルでは d²T/dx² = 0 → flux = 0
        inner = f[1:-1]
        np.testing.assert_allclose(inner, 0.0, atol=1e-12)

    def test_matrix_row_sum_zero_interior(self):
        """内部セルでは行列の行和がゼロ（保存性）."""
        mesh_result = StructuredMeshProcess().process(
            StructuredMeshInput(Lx=1.0, Ly=1.0, Lz=1.0, nx=5, ny=5, nz=5)
        )
        mesh = mesh_result.mesh
        scheme = CentralDiffusionScheme()
        A = scheme.matrix_coefficients(1.0, mesh)

        row_sums = np.array(A.sum(axis=1)).ravel()
        # 境界面を持たないセルは行和が0
        # 構造化格子の内部セル: i in [1,nx-2], j in [1,ny-2], k in [1,nz-2]
        nx, ny, nz = mesh.dimensions
        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                for k in range(1, nz - 1):
                    idx = i * ny * nz + j * nz + k
                    assert abs(row_sums[idx]) < 1e-14, (
                        f"cell ({i},{j},{k}): row_sum={row_sums[idx]}"
                    )

    def test_heterogeneous_diffusivity(self):
        """不均一拡散係数で行列が対称かつ正定値的."""
        nx = 4
        mesh_result = StructuredMeshProcess().process(
            StructuredMeshInput(Lx=1.0, Ly=1.0, Lz=1.0, nx=nx, ny=nx, nz=nx)
        )
        mesh = mesh_result.mesh
        # ランダムな正の拡散係数
        rng = np.random.default_rng(42)
        gamma = rng.uniform(0.1, 10.0, mesh.n_cells)

        scheme = CentralDiffusionScheme()
        A = scheme.matrix_coefficients(gamma, mesh)

        # 対称性
        diff = A - A.T
        assert diff.nnz == 0 or np.max(np.abs(diff.data)) < 1e-12

        # 対角成分は正
        diag = A.diagonal()
        assert np.all(diag >= 0)

    def test_diffusivity_scaling(self):
        """拡散係数を2倍にすると行列要素も2倍."""
        mesh_result = StructuredMeshProcess().process(
            StructuredMeshInput(Lx=1.0, Ly=1.0, Lz=1.0, nx=3, ny=3, nz=3)
        )
        scheme = CentralDiffusionScheme()
        A1 = scheme.matrix_coefficients(1.0, mesh_result.mesh)
        A2 = scheme.matrix_coefficients(2.0, mesh_result.mesh)
        np.testing.assert_allclose(A2.toarray(), 2.0 * A1.toarray(), rtol=1e-14)

    def test_stretched_mesh(self):
        """不等間隔格子でも対称行列."""
        mesh_result = StructuredMeshProcess().process(
            StructuredMeshInput(
                Lx=1.0,
                Ly=1.0,
                Lz=1.0,
                nx=5,
                ny=5,
                nz=5,
                stretch_x=(3.0, 1.0),
            )
        )
        scheme = CentralDiffusionScheme()
        A = scheme.matrix_coefficients(1.0, mesh_result.mesh)
        diff = A - A.T
        assert diff.nnz == 0 or np.max(np.abs(diff.data)) < 1e-14
