"""自然対流シミュレーション調査スクリプト.

2次元直交格子の過渡熱流体モデルを使い、中心発熱源からの自然対流の
発達度合いを、ドメインサイズ・発熱量・境界条件の観点から調査する。

境界条件パターン:
  A: 密閉キャビティ（全辺 NO_SLIP 壁）
  B: 半開放（左右壁、下 INLET 微小速度、上 OUTLET_PRESSURE）
  C: 3辺開放（左右+上 OUTLET_PRESSURE、下 INLET 微小速度）

空気物性(300K近辺)を使用。
"""

from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from xkep_cae_fluid.natural_convection.data import (
    FluidBoundaryCondition,
    FluidBoundarySpec,
    NaturalConvectionInput,
    ThermalBoundaryCondition,
)
from xkep_cae_fluid.natural_convection.solver import NaturalConvectionFDMProcess

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 空気物性 (300K)
# ---------------------------------------------------------------------------
AIR_RHO = 1.18  # kg/m³
AIR_MU = 1.85e-5  # Pa·s
AIR_CP = 1007.0  # J/(kg·K)
AIR_K = 0.026  # W/(m·K)
AIR_BETA = 1.0 / 300.0  # 1/K (理想気体近似)
AIR_NU = AIR_MU / AIR_RHO  # m²/s
AIR_ALPHA = AIR_K / (AIR_RHO * AIR_CP)  # m²/s

T_REF = 300.0  # K
HEATER_SIZE = 0.01  # m (発熱体サイズ固定)


@dataclass
class CaseResult:
    """1ケースの結果."""

    bc_pattern: str
    domain_size: float
    q_vol: float
    converged: bool
    n_iterations: int
    elapsed_s: float
    v_max: float
    delta_T_max: float
    ra_star: float
    mass_residual: float
    T_center: float


def _build_bc_pattern_a(
    T_ref: float,
) -> dict[str, FluidBoundarySpec]:
    """密閉キャビティ: 全辺 NO_SLIP + DIRICHLET温度."""
    wall_cold = FluidBoundarySpec(
        condition=FluidBoundaryCondition.NO_SLIP,
        thermal=ThermalBoundaryCondition.DIRICHLET,
        temperature=T_ref,
    )
    return {
        "bc_xm": wall_cold,
        "bc_xp": wall_cold,
        "bc_ym": wall_cold,
        "bc_yp": wall_cold,
    }


def _build_bc_pattern_b(T_ref: float, v_inlet: float = 0.001) -> dict[str, FluidBoundarySpec]:
    """半開放: 左右壁、下 INLET、上 OUTLET_PRESSURE."""
    wall = FluidBoundarySpec(
        condition=FluidBoundaryCondition.NO_SLIP,
        thermal=ThermalBoundaryCondition.DIRICHLET,
        temperature=T_ref,
    )
    inlet = FluidBoundarySpec(
        condition=FluidBoundaryCondition.INLET_VELOCITY,
        velocity=(0.0, v_inlet, 0.0),
        thermal=ThermalBoundaryCondition.DIRICHLET,
        temperature=T_ref,
    )
    outlet = FluidBoundarySpec(
        condition=FluidBoundaryCondition.OUTLET_PRESSURE,
        pressure=0.0,
        thermal=ThermalBoundaryCondition.DIRICHLET,
        temperature=T_ref,
    )
    return {
        "bc_xm": wall,
        "bc_xp": wall,
        "bc_ym": inlet,
        "bc_yp": outlet,
    }


def _build_bc_pattern_c(T_ref: float, v_inlet: float = 0.001) -> dict[str, FluidBoundarySpec]:
    """3辺開放: 左右+上 OUTLET_PRESSURE、下 INLET."""
    outlet = FluidBoundarySpec(
        condition=FluidBoundaryCondition.OUTLET_PRESSURE,
        pressure=0.0,
        thermal=ThermalBoundaryCondition.DIRICHLET,
        temperature=T_ref,
    )
    inlet = FluidBoundarySpec(
        condition=FluidBoundaryCondition.INLET_VELOCITY,
        velocity=(0.0, v_inlet, 0.0),
        thermal=ThermalBoundaryCondition.DIRICHLET,
        temperature=T_ref,
    )
    return {
        "bc_xm": outlet,
        "bc_xp": outlet,
        "bc_ym": inlet,
        "bc_yp": outlet,
    }


BC_BUILDERS = {
    "A_closed": _build_bc_pattern_a,
    "B_semi_open": _build_bc_pattern_b,
    "C_three_open": _build_bc_pattern_c,
}


def run_case(
    bc_pattern: str,
    L: float,
    q: float,
    nx: int = 30,
    ny: int = 30,
) -> CaseResult:
    """1ケースを実行."""
    nz = 1
    Lz = L / nx  # 2D: z方向1セル分

    # 発熱体セル範囲（中心付近）
    heater_cells = max(1, int(HEATER_SIZE / (L / nx)))
    i_start = nx // 2 - heater_cells // 2
    i_end = i_start + heater_cells
    j_start = ny // 2 - heater_cells // 2
    j_end = j_start + heater_cells

    q_vol = np.zeros((nx, ny, nz))
    q_vol[i_start:i_end, j_start:j_end, :] = q

    # BC構築
    bc_builder = BC_BUILDERS[bc_pattern]
    bcs = bc_builder(T_REF)

    # z方向: SYMMETRY
    z_bc = FluidBoundarySpec(
        condition=FluidBoundaryCondition.SYMMETRY,
        thermal=ThermalBoundaryCondition.ADIABATIC,
    )

    # 修正Ra数: Ra* = g*beta*q*L^5 / (k*nu*alpha)
    ra_star = 9.81 * AIR_BETA * q * L**5 / (AIR_K * AIR_NU * AIR_ALPHA)

    logger.info(
        "Case: bc=%s, L=%.3f, q=%.0f, Ra*=%.2e, heater=%dx%d cells",
        bc_pattern,
        L,
        q,
        ra_star,
        heater_cells,
        heater_cells,
    )

    inp = NaturalConvectionInput(
        Lx=L,
        Ly=L,
        Lz=Lz,
        nx=nx,
        ny=ny,
        nz=nz,
        rho=AIR_RHO,
        mu=AIR_MU,
        Cp=AIR_CP,
        k_fluid=AIR_K,
        beta=AIR_BETA,
        T_ref=T_REF,
        gravity=(0.0, -9.81, 0.0),
        q_vol=q_vol,
        bc_xm=bcs["bc_xm"],
        bc_xp=bcs["bc_xp"],
        bc_ym=bcs["bc_ym"],
        bc_yp=bcs["bc_yp"],
        bc_zm=z_bc,
        bc_zp=z_bc,
        max_simple_iter=2000,
        tol_simple=1e-4,
        alpha_u=0.2,
        alpha_p=0.05,
        alpha_T=0.5,
    )

    solver = NaturalConvectionFDMProcess()
    result = solver.process(inp)

    v_max = float(max(np.abs(result.u).max(), np.abs(result.v).max()))
    delta_T_max = float(result.T.max() - T_REF)
    T_center = float(result.T[nx // 2, ny // 2, 0])
    mass_res = (
        float(result.residual_history["mass"][-1])
        if result.residual_history.get("mass")
        else float("nan")
    )

    logger.info(
        "  Result: converged=%s, iter=%d, v_max=%.4e, dT_max=%.2f, T_center=%.2f, elapsed=%.1fs",
        result.converged,
        result.n_outer_iterations,
        v_max,
        delta_T_max,
        T_center,
        result.elapsed_seconds,
    )

    return CaseResult(
        bc_pattern=bc_pattern,
        domain_size=L,
        q_vol=q,
        converged=result.converged,
        n_iterations=result.n_outer_iterations,
        elapsed_s=round(result.elapsed_seconds, 2),
        v_max=round(v_max, 6),
        delta_T_max=round(delta_T_max, 4),
        ra_star=round(ra_star, 2),
        mass_residual=round(mass_res, 8),
        T_center=round(T_center, 4),
    )


def main():
    """パラメトリックスタディを実行."""
    domain_sizes = [0.05, 0.1, 0.2]
    q_values = [1000.0, 5000.0, 10000.0]
    bc_patterns = ["A_closed", "B_semi_open", "C_three_open"]

    results: list[dict] = []
    total = len(domain_sizes) * len(q_values) * len(bc_patterns)
    count = 0

    t_start = time.perf_counter()

    for L in domain_sizes:
        for q in q_values:
            for bc in bc_patterns:
                count += 1
                logger.info("=" * 60)
                logger.info("Case %d/%d", count, total)

                try:
                    cr = run_case(bc, L, q)
                    results.append(
                        {
                            "bc_pattern": cr.bc_pattern,
                            "domain_size_m": cr.domain_size,
                            "q_vol_W_m3": cr.q_vol,
                            "converged": cr.converged,
                            "n_iterations": cr.n_iterations,
                            "elapsed_s": cr.elapsed_s,
                            "v_max_m_s": cr.v_max,
                            "delta_T_max_K": cr.delta_T_max,
                            "ra_star": cr.ra_star,
                            "mass_residual": cr.mass_residual,
                            "T_center_K": cr.T_center,
                        }
                    )
                except Exception as e:
                    logger.error("Case failed: %s", e)
                    results.append(
                        {
                            "bc_pattern": bc,
                            "domain_size_m": L,
                            "q_vol_W_m3": q,
                            "converged": False,
                            "error": str(e),
                        }
                    )

    total_time = time.perf_counter() - t_start

    # 結果出力
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "natural_convection_investigation.yaml"

    output_data = {
        "investigation": "natural_convection_parametric_study",
        "date": "2026-04-01",
        "air_properties_300K": {
            "rho_kg_m3": AIR_RHO,
            "mu_Pa_s": AIR_MU,
            "Cp_J_kgK": AIR_CP,
            "k_W_mK": AIR_K,
            "beta_1_K": round(AIR_BETA, 6),
            "nu_m2_s": round(AIR_NU, 8),
            "alpha_m2_s": round(AIR_ALPHA, 8),
            "Pr": round(AIR_NU / AIR_ALPHA, 4),
        },
        "heater_size_m": HEATER_SIZE,
        "mesh": "nx=ny=30, nz=1 (2D)",
        "solver": {
            "alpha_u": 0.2,
            "alpha_p": 0.05,
            "alpha_T": 0.5,
            "max_simple_iter": 2000,
            "tol_simple": 1e-4,
        },
        "total_elapsed_s": round(total_time, 1),
        "results": results,
    }

    with open(output_path, "w") as f:
        yaml.dump(output_data, f, default_flow_style=False, allow_unicode=True)

    logger.info("=" * 60)
    logger.info("Total elapsed: %.1f s", total_time)
    logger.info("Results written to: %s", output_path)

    # サマリー表出力
    print("\n" + "=" * 100)
    print(
        f"{'BC':>15} {'L(m)':>6} {'q(W/m3)':>8} {'Ra*':>12} "
        f"{'conv':>5} {'iter':>5} {'v_max':>10} {'dT_max(K)':>10} "
        f"{'T_center':>10}"
    )
    print("-" * 100)
    for r in results:
        if "error" in r:
            print(
                f"{r['bc_pattern']:>15} {r['domain_size_m']:>6.3f} "
                f"{r['q_vol_W_m3']:>8.0f} {'ERROR':>12}"
            )
        else:
            print(
                f"{r['bc_pattern']:>15} {r['domain_size_m']:>6.3f} "
                f"{r['q_vol_W_m3']:>8.0f} {r['ra_star']:>12.2e} "
                f"{str(r['converged']):>5} {r['n_iterations']:>5} "
                f"{r['v_max_m_s']:>10.4e} {r['delta_T_max_K']:>10.4f} "
                f"{r['T_center_K']:>10.4f}"
            )
    print("=" * 100)

    return 0


if __name__ == "__main__":
    sys.exit(main())
