"""中心差分拡散スキーム（MeshData ベース FVM 離散化）.

直交格子上の中心差分法による拡散項離散化を提供する。
DiffusionSchemeStrategy Protocol の具象実装。

面フラックス: F_diff = Γ_f * A_f * (φ_N - φ_P) / d_PN

ここで:
  Γ_f: 面における拡散係数（隣接セルの調和平均）
  A_f: 面面積
  φ_P, φ_N: owner セルおよび neighbour セルのスカラー値
  d_PN: セル中心間距離
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from xkep_cae_fluid.core.data import MeshData


class CentralDiffusionScheme:
    """中心差分拡散スキーム.

    MeshData の面情報を使い、拡散項の係数行列とフラックスを構築する。
    直交格子に対して2次精度。
    """

    def flux(
        self,
        phi: np.ndarray,
        diffusivity: float | np.ndarray,
        mesh: MeshData,
    ) -> np.ndarray:
        """拡散フラックスを計算.

        各セルに対し、隣接面を通じた拡散フラックスの合計を返す。

        Parameters
        ----------
        phi : np.ndarray
            スカラー場 (n_cells,)
        diffusivity : float | np.ndarray
            拡散係数。スカラーまたはセルごとの配列 (n_cells,)
        mesh : MeshData
            メッシュデータ

        Returns
        -------
        np.ndarray
            拡散フラックス (n_cells,)。正値は流入を意味する。
        """
        n_cells = mesh.n_cells
        owner = mesh.face_owner
        neighbour = mesh.face_neighbour
        n_internal = len(neighbour)

        # 面ごとの拡散係数（調和平均）
        gamma_f = self._face_diffusivity(diffusivity, owner, neighbour, n_internal)

        # セル中心間距離
        d_pn = self._face_distance(mesh, n_internal)

        # 面面積
        areas = mesh.face_areas[:n_internal]

        # 面フラックス: Γ_f * A_f * (φ_N - φ_P) / d_PN
        dphi = phi[neighbour] - phi[owner[:n_internal]]
        face_flux = gamma_f * areas * dphi / d_pn

        # セルごとの合計
        result = np.zeros(n_cells)
        np.add.at(result, owner[:n_internal], face_flux)
        np.add.at(result, neighbour, -face_flux)

        return result

    def matrix_coefficients(
        self,
        diffusivity: float | np.ndarray,
        mesh: MeshData,
    ) -> sp.csr_matrix:
        """拡散項の係数行列を構築.

        A_diff * φ = 0 の形式で、拡散項を表現する疎行列を返す。
        対角成分は正、非対角成分は負（SPD行列）。

        Parameters
        ----------
        diffusivity : float | np.ndarray
            拡散係数。スカラーまたはセルごとの配列 (n_cells,)
        mesh : MeshData
            メッシュデータ

        Returns
        -------
        sp.csr_matrix
            拡散係数行列 (n_cells, n_cells)
        """
        n_cells = mesh.n_cells
        owner = mesh.face_owner
        neighbour = mesh.face_neighbour
        n_internal = len(neighbour)

        # 面ごとの拡散係数（調和平均）
        gamma_f = self._face_diffusivity(diffusivity, owner, neighbour, n_internal)

        # セル中心間距離
        d_pn = self._face_distance(mesh, n_internal)

        # 面面積
        areas = mesh.face_areas[:n_internal]

        # 面係数: a_f = Γ_f * A_f / d_PN
        a_f = gamma_f * areas / d_pn

        # COO 形式で組立
        # 非対角: owner-neighbour 間
        row = np.concatenate([owner[:n_internal], neighbour])
        col = np.concatenate([neighbour, owner[:n_internal]])
        data = np.concatenate([-a_f, -a_f])

        # 対角: 各セルの面係数の合計
        diag = np.zeros(n_cells)
        np.add.at(diag, owner[:n_internal], a_f)
        np.add.at(diag, neighbour, a_f)

        row = np.concatenate([row, np.arange(n_cells)])
        col = np.concatenate([col, np.arange(n_cells)])
        data = np.concatenate([data, diag])

        return sp.csr_matrix((data, (row, col)), shape=(n_cells, n_cells))

    @staticmethod
    def _face_diffusivity(
        diffusivity: float | np.ndarray,
        owner: np.ndarray,
        neighbour: np.ndarray,
        n_internal: int,
    ) -> np.ndarray:
        """面における拡散係数（調和平均）を計算."""
        if np.isscalar(diffusivity):
            return np.full(n_internal, float(diffusivity))

        gamma_p = diffusivity[owner[:n_internal]]
        gamma_n = diffusivity[neighbour]
        # 調和平均: 2 * Γ_P * Γ_N / (Γ_P + Γ_N)
        denom = gamma_p + gamma_n
        # ゼロ除算回避
        safe = denom > 0
        result = np.zeros(n_internal)
        result[safe] = 2.0 * gamma_p[safe] * gamma_n[safe] / denom[safe]
        return result

    @staticmethod
    def _face_distance(mesh: MeshData, n_internal: int) -> np.ndarray:
        """内部面に対するセル中心間距離を計算."""
        owner = mesh.face_owner[:n_internal]
        neighbour = mesh.face_neighbour

        cc_owner = mesh.cell_centers[owner]
        cc_neighbour = mesh.cell_centers[neighbour]
        delta = cc_neighbour - cc_owner
        return np.linalg.norm(delta, axis=1)
