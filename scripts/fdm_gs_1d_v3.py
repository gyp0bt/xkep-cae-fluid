"""1D過渡固体電熱解析スクリプト (Gauss-Seidel)

区間ごとに物性値と断面積が異なる1次元ロッドのジュール加熱を
制御体積法 + 陰的Euler + Gauss-Seidel反復で解く。

物理:
    rho * c * A * dT/dt = d/dx [k * A * dT/dx] + I^2 * rho_e(T) / A

    rho_e(T) = rho_e0 * (1 + alpha * (T - T_ref))
    q_vol = I^2 * rho_e(T) / A^2  [W/m^3]

離散化:
    セル中心有限体積法、陰的Euler時間積分
    面コンダクタンスは直列抵抗モデル (熱抵抗の直列接続):
        1/C_{i+1/2} = dx_i/(2*k_i*A_i) + dx_{i+1}/(2*k_{i+1}*A_{i+1})

"""

from dataclasses import dataclass

import numpy as np


def solve_1d_joule_gs(
    # ---- メッシュ ----
    n_cells_per_seg: list[int],  # 各区間のセル数
    seg_lengths: list[float],  # 各区間の長さ [m]
    seg_areas: list[float],  # 各区間の断面積 [m^2]
    seg_areas_conv: list[float],  # 各区間の放熱面積 [m^2]
    seg_h: list[float],  # 各区間の熱伝達係数 [W/m^2K]
    # ---- 物性 (区間ごと) ----
    seg_rho: list[float],  # 密度 [kg/m^3]
    seg_c: list[float],  # 比熱 [J/(kg*K)]
    seg_k: list[float],  # 熱伝導率 [W/(m*K)]
    seg_rho_e0: list[float],  # 基準電気抵抗率 [Ohm*m]
    seg_alpha: list[float],  # 抵抗温度係数 [1/K]
    # ---- 輻射 (区間ごと) ----
    seg_epsilon: list[float] | None = None,  # 輻射率 [-]
    seg_areas_rad: list[float] | None = None,  # 輻射面積 [m^2] (Noneで放熱面積と同じ)
    # ---- 条件 ----
    current_I: float = 0.0,  # 電流 [A]
    T_init: float = 300.0,  # 初期温度 [K]
    T_env: float = 300.0,  # 環境温度 [K]
    T_ref: float = 300.0,  # 抵抗の基準温度 [K]
    dt: float = 1e-3,  # 時間刻み [s]
    t_end: float = 1.0,  # 終了時刻 [s]
    T_left: float | None = None,  # 左境界温度 [K]
    T_right: float | None = None,  # 右境界温度 [K]
    T_melt: float | None = None,  # 溶断温度 [K] (超えたら停止)
    # ---- ソルバー ----
    gs_tol: float = 1e-10,
    gs_max_iter: int = 500,
    outer_max_iter: int = 10,
    print_every: int = 0,
):
    """1D過渡ジュール電熱をGauss-Seidelで解く。

    Returns: (T, x_center, t_final, info_dict)
    """
    n_seg = len(seg_lengths)

    # ---- デフォルト値の補完 ----
    if seg_epsilon is None:
        seg_epsilon = [0.0] * n_seg  # 輻射なし
    if seg_areas_rad is None:
        seg_areas_rad = list(seg_areas_conv)  # 放熱面積と同じ

    SIGMA = 5.670374419e-8  # Stefan-Boltzmann 定数

    # ---- セル配列の構築 ----
    dx_list, rho_list, c_list, k_list, A_list, A_conv_list, seg_h_list = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    rho_e0_list, alpha_list = [], []
    eps_list, A_rad_list = [], []
    for s in range(n_seg):
        cell_dx = seg_lengths[s] / n_cells_per_seg[s]
        for _ in range(n_cells_per_seg[s]):
            dx_list.append(cell_dx)
            rho_list.append(seg_rho[s])
            c_list.append(seg_c[s])
            k_list.append(seg_k[s])
            A_list.append(seg_areas[s])
            A_conv_list.append(seg_areas_conv[s] / n_cells_per_seg[s])
            seg_h_list.append(seg_h[s])
            rho_e0_list.append(seg_rho_e0[s])
            alpha_list.append(seg_alpha[s])
            eps_list.append(seg_epsilon[s])
            A_rad_list.append(seg_areas_rad[s] / n_cells_per_seg[s])

    N = len(dx_list)
    dx = np.array(dx_list)
    rho = np.array(rho_list)
    c = np.array(c_list)
    k = np.array(k_list)
    A = np.array(A_list)
    A_conv = np.array(A_conv_list)
    h = np.array(seg_h_list)
    rho_e0 = np.array(rho_e0_list)
    alpha_e = np.array(alpha_list)
    eps = np.array(eps_list)
    A_rad = np.array(A_rad_list)

    # セル中心座標
    x_center = np.zeros(N)
    x_center[0] = dx[0] / 2.0
    for i in range(1, N):
        x_center[i] = x_center[i - 1] + dx[i - 1] / 2.0 + dx[i] / 2.0

    # ---- 面コンダクタンス (直列抵抗モデル) ----
    # 内部面 i+1/2 (i=0..N-2):
    #   1/C = dx_i/(2*k_i*A_i) + dx_{i+1}/(2*k_{i+1}*A_{i+1})
    C_face = np.zeros(N - 1)
    for i in range(N - 1):
        R = dx[i] / (2.0 * k[i] * A[i]) + dx[i + 1] / (2.0 * k[i + 1] * A[i + 1])
        C_face[i] = 1.0 / R

    # 境界面コンダクタンス (半セル距離)
    C_left_bnd = k[0] * A[0] / (dx[0] / 2.0)
    C_right_bnd = k[-1] * A[-1] / (dx[-1] / 2.0)

    # ---- 係数 (時間不変部分) ----
    # 過渡項: a_P0 = rho * c * A * dx / dt
    V = A * dx  # セル体積
    # a_P0 = rho * c * V / dt はループ内で this_dt に応じて再計算

    # 西側・東側係数
    a_W = np.zeros(N)
    a_E = np.zeros(N)
    for i in range(N):
        if i > 0:
            a_W[i] = C_face[i - 1]
        if i < N - 1:
            a_E[i] = C_face[i]

    # 境界寄与 (a_Pへの加算分と、bへの定数項)
    # T_left/T_right が None の場合は断熱 (ゼロ勾配) → 境界コンダクタンス不要
    a_P_bnd = np.zeros(N)
    b_bnd = np.zeros(N)
    if T_left is not None:
        a_P_bnd[0] = C_left_bnd
        b_bnd[0] = C_left_bnd * T_left
    if T_right is not None:
        a_P_bnd[-1] = C_right_bnd
        b_bnd[-1] = C_right_bnd * T_right

    # ---- 時間進行 ----
    T = np.full(N, T_init)
    t = 0.0
    step = 0
    melted = False

    while t < t_end - 1e-15:
        this_dt = min(dt, t_end - t)
        # a_P0を現在のdtで再計算 (最終ステップでdtが変わる場合)
        # a_P0_now = rho * c * V / this_dt
        a_P0_now = rho * c * V / this_dt
        a_P1_now = h * A_conv
        T_old = T.copy()

        # 外側反復 (非線形ソース項の再評価)
        for _outer in range(outer_max_iter):
            T_star = T.copy()

            # ジュール発熱: q_vol = I^2 * rho_e(T*) / A^2
            rho_e = rho_e0 * (1.0 + alpha_e * (T_star - T_ref))
            source = current_I**2 * rho_e / A**2 * V  # [W] per cell

            # 輻射線形化係数: h_rad = ε·σ·(T*² + T_env²)·(T* + T_env)
            # q_rad = h_rad · A_rad · (T - T_env) と近似
            h_rad = eps * SIGMA * (T_star**2 + T_env**2) * (T_star + T_env)
            a_rad = h_rad * A_rad  # [W/K] per cell

            # Gauss-Seidel 内側反復
            converged = False
            for _gs in range(gs_max_iter):
                T_prev = T.copy()
                for i in range(N):
                    a_P = a_P0_now[i] + a_P1_now[i] + a_rad[i] + a_W[i] + a_E[i] + a_P_bnd[i]
                    b_i = (
                        a_P0_now[i] * T_old[i]
                        + (a_P1_now[i] + a_rad[i]) * T_env
                        + source[i]
                        + b_bnd[i]
                    )
                    if i > 0:
                        b_i += a_W[i] * T[i - 1]
                    if i < N - 1:
                        b_i += a_E[i] * T[i + 1]
                    T[i] = b_i / a_P

                if np.max(np.abs(T - T_prev)) < gs_tol:
                    converged = True
                    break

            # 外側収束判定
            if np.max(np.abs(T - T_star)) < gs_tol:
                break

        t += this_dt
        step += 1

        if print_every > 0 and step % print_every == 0:
            print(f"  t={t:.6e}s  step={step}  T_max={np.max(T):.2f}K  conv={converged}")

        if T_melt is not None and np.max(T) >= T_melt:
            melted = True
            if print_every > 0:
                print(f"  ** 溶断温度 {T_melt}K 到達 at t={t:.6e}s **")
            break

    # ---- エネルギー収支 ----
    rho_e_final = rho_e0 * (1.0 + alpha_e * (T - T_ref))
    q_vol_final = current_I**2 * rho_e_final / A**2
    Q_joule = np.sum(q_vol_final * V)
    Q_left_out = C_left_bnd * (T[0] - T_left) if T_left is not None else 0.0
    Q_right_out = C_right_bnd * (T[-1] - T_right) if T_right is not None else 0.0
    Q_conv = np.sum(h * A_conv * (T - T_env))
    Q_rad = np.sum(eps * SIGMA * A_rad * (T**4 - T_env**4))
    balance_err = Q_joule - Q_left_out - Q_right_out - Q_conv - Q_rad

    info = {
        "t_final": t,
        "steps": step,
        "melted": melted,
        "T_max": np.max(T),
        "T_max_pos": x_center[np.argmax(T)],
        "Q_joule": Q_joule,
        "Q_left": Q_left_out,
        "Q_right": Q_right_out,
        "Q_conv": Q_conv,
        "Q_rad": Q_rad,
        "balance_err": balance_err,
        "dx": dx,
        "k": k,
        "A": A,
        "V": V,
        "N": N,
    }
    return T, x_center, info


def analytical_steady_uniform(x, L, T_bc, current, rho_e0, k, A):
    """均一ロッド定常解 (alpha=0): T(x) = T_bc + I²ρ_e0/(2kA²) * x(L-x)"""
    return T_bc + (current**2 * rho_e0) / (2.0 * k * A**2) * x * (L - x)


@dataclass
class LineArea:
    """1D電熱解析の区間定義 (PCBトレース/ワイヤ等)"""

    # 電流 [A]
    current: float = 5.0

    # 材料物性
    k: float = 386.0  # 熱伝導率 [W/(m·K)]
    density: float = 8950  # 密度 [kg/m³]
    c: float = 380.0  # 比熱 [J/(kg·K)]
    rho: float = 1.68e-8  # 基準電気抵抗率 [Ω·m]
    alpha: float = 3.93e-3  # 抵抗温度係数 [1/K]
    T_m: float = 1085.0  # 融点 [K]
    C_l: float | None = None  # 長さ当たり熱容量 [J/(K·m)]

    # パターン幾何
    length: float = 7.5e-3  # パターン長 [m]
    A0: float = 0.0035e-6  # 発熱断面積 [m²]
    A1: float = 1.36e-4 * 7.5e-3 * 2.0  # 放熱面積 [m²]
    Rth_l: float | None = None  # 長さ当たり熱抵抗 [K·m/W]

    # 輻射
    epsilon: float = 0.1  # 輻射率 [-]
    A_rad: float | None = None  # 輻射面積 [m²] (None=A1と同じ)

    # 環境条件
    T_env: float = 25.0  # 環境温度 [℃]
    h_env: float = 5.0  # 対流熱伝達係数 [W/(m²·K)]

    # セル分割数 (デフォルト=10)
    n_cells: int = 10


def solve_from_line_areas(
    segments: list[LineArea],
    *,
    T_init: float = 300.0,
    T_env: float | None = None,
    T_ref: float = 300.0,
    dt: float = 1e-4,
    t_end: float = 1.0,
    T_left: float | None = None,
    T_right: float | None = None,
    T_melt: float | None = None,
    gs_tol: float = 1e-10,
    gs_max_iter: int = 500,
    outer_max_iter: int = 10,
    print_every: int = 0,
):
    """LineAreaリストからsolve_1d_joule_gsを呼び出すヘルパー。

    各 LineArea が1区間に対応し、直列接続として解析する。
    電流は先頭セグメントの値を使用 (直列なので全区間同一)。

    Args:
        segments: LineArea のリスト (直列接続順)
        T_init: 初期温度 [K]
        T_env: 環境温度 [K] (None の場合、最初のセグメントの T_env+273.15)
        T_ref: 抵抗基準温度 [K]
        dt, t_end: 時間刻み・終了時刻
        T_left, T_right: 左右境界温度 (None=断熱)
        T_melt: 溶断温度 [K]

    Returns:
        (T, x_center, info)
    """
    if not segments:
        raise ValueError("segments は空にできません")

    current_I = segments[0].current

    # 環境温度: LineArea は℃で定義 → K に変換
    if T_env is None:
        T_env_K = segments[0].T_env + 273.15
    else:
        T_env_K = T_env

    # 各区間パラメータの展開
    n_cells_per_seg = []
    seg_lengths = []
    seg_areas = []
    seg_areas_conv = []
    seg_h = []
    seg_rho = []
    seg_c = []
    seg_k = []
    seg_rho_e0 = []
    seg_alpha = []
    seg_epsilon = []
    seg_areas_rad = []

    for seg in segments:
        n_cells_per_seg.append(seg.n_cells)
        seg_lengths.append(seg.length)
        seg_areas.append(seg.A0)
        seg_areas_conv.append(seg.A1)
        seg_h.append(seg.h_env)
        seg_rho.append(seg.density)
        seg_c.append(seg.c)
        seg_k.append(seg.k)
        seg_rho_e0.append(seg.rho)
        seg_alpha.append(seg.alpha)
        seg_epsilon.append(seg.epsilon)
        # 輻射面積: 明示されていなければ放熱面積と同じ
        seg_areas_rad.append(seg.A_rad if seg.A_rad is not None else seg.A1)

    return solve_1d_joule_gs(
        n_cells_per_seg=n_cells_per_seg,
        seg_lengths=seg_lengths,
        seg_areas=seg_areas,
        seg_areas_conv=seg_areas_conv,
        seg_h=seg_h,
        seg_rho=seg_rho,
        seg_c=seg_c,
        seg_k=seg_k,
        seg_rho_e0=seg_rho_e0,
        seg_alpha=seg_alpha,
        seg_epsilon=seg_epsilon,
        seg_areas_rad=seg_areas_rad,
        current_I=current_I,
        T_init=T_init,
        T_env=T_env_K,
        T_ref=T_ref,
        dt=dt,
        t_end=t_end,
        T_left=T_left,
        T_right=T_right,
        T_melt=T_melt,
        gs_tol=gs_tol,
        gs_max_iter=gs_max_iter,
        outer_max_iter=outer_max_iter,
        print_every=print_every,
    )


def _print_result(T, xc, info, n_show=15):
    """結果の表示ヘルパー"""
    print(f"  最終時刻   : {info['t_final']:.6e} s")
    print(f"  溶断到達   : {info['melted']}")
    print(f"  T_max      : {info['T_max']:.2f} K  at x = {info['T_max_pos'] * 1000:.2f} mm")
    print(f"  ステップ数 : {info['steps']}")
    print(f"  Q_joule    : {info['Q_joule']:.6e} W")
    print(f"  Q_conv     : {info['Q_conv']:.6e} W")
    print(f"  Q_rad      : {info['Q_rad']:.6e} W")
    print(f"  Q_left     : {info['Q_left']:.6e} W")
    print(f"  Q_right    : {info['Q_right']:.6e} W")
    print(f"  balance_err: {info['balance_err']:.6e} W")
    print()
    print("  温度プロファイル:")
    n_show = min(len(T), n_show)
    indices = np.linspace(0, len(T) - 1, n_show, dtype=int)
    for idx in indices:
        bar = "#" * max(0, int((T[idx] - 300) / 10))
        print(f"    x={xc[idx] * 1000:6.3f}mm  T={T[idx]:8.2f}K  {bar}")
    print()


if __name__ == "__main__":
    # ================================================================
    # Demo 1: 断熱境界 + ジュール加熱 → 均一昇温の確認
    # ================================================================
    print("=" * 70)
    print("Demo 1: 断熱境界 + ジュール加熱 (均一昇温確認)")
    print("=" * 70)
    seg = LineArea(
        current=5.0,
        k=11.3,
        density=8400.0,
        c=450.0,
        rho=1.1e-6,
        alpha=4e-4,
        length=0.01,
        A0=1e-8,
        A1=1e-6,
        h_env=5.0,
        epsilon=0.0,  # 輻射なし
        n_cells=10,
    )
    T, xc, info = solve_from_line_areas(
        [seg],
        T_init=300.0,
        T_env=300.0,
        T_ref=300.0,
        dt=1e-5,
        t_end=0.05,
        T_melt=1000.0,
        print_every=500,
    )
    _print_result(T, xc, info)

    # ================================================================
    # Demo 2: 輻射あり → 定常温度の確認
    # ================================================================
    print("=" * 70)
    print("Demo 2: 輻射あり定常解析 (対流+輻射バランス)")
    print("=" * 70)
    seg_rad = LineArea(
        current=5.0,
        k=11.3,
        density=8400.0,
        c=450.0,
        rho=1.1e-6,
        alpha=4e-4,
        length=0.01,
        A0=1e-8,
        A1=1e-5,
        h_env=5.0,
        epsilon=0.8,  # 高輻射率
        n_cells=20,
    )
    T, xc, info = solve_from_line_areas(
        [seg_rad],
        T_init=300.0,
        T_env=300.0,
        T_ref=300.0,
        dt=1e-4,
        t_end=1.0,
        print_every=500,
    )
    _print_result(T, xc, info)

    # ================================================================
    # Demo 3: 銅-ニクロム-銅 の3区間直列 + 輻射
    # ================================================================
    print("=" * 70)
    print("Demo 3: 銅-ニクロム-銅 3区間直列 + 輻射")
    print("=" * 70)
    cu = LineArea(
        current=5.0,
        k=386.0,
        density=8950.0,
        c=380.0,
        rho=1.68e-8,
        alpha=3.93e-3,
        length=5e-3,
        A0=1e-7,
        A1=1e-5,
        h_env=5.0,
        epsilon=0.3,
        n_cells=10,
    )
    nichrome = LineArea(
        current=5.0,
        k=11.3,
        density=8400.0,
        c=450.0,
        rho=1.1e-6,
        alpha=4e-4,
        length=10e-3,
        A0=1e-7,
        A1=1e-5,
        h_env=5.0,
        epsilon=0.8,
        n_cells=20,
    )
    T, xc, info = solve_from_line_areas(
        [cu, nichrome, cu],
        T_init=300.0,
        T_env=300.0,
        T_ref=300.0,
        dt=1e-4,
        t_end=5.0,
        T_left=300.0,
        T_right=300.0,
        print_every=2000,
    )
    _print_result(T, xc, info)
