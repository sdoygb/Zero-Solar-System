# -*- coding: utf-8 -*-
"""太阳系轨道互动页（3D）的数据生成、核验与构建。

用法：python3 web/gen_data.py

流程：
  1. 重算页面里用到的每一个数（行星表、壳层判据、地月终态、太阳偏置、
     96^3 周期格上的长程力），并与篇目里已登记的值对照；
  2. 读 index.src.html，把数据与离线的 three.js 注入两个标记块，写出 index.html；
  3. 打印核验表，任何一项超出容差即以非零码退出。

依据：Zb012（离散泊松与长程势）、Zb050（分栏与壳层判据）、
      Zb051（开普勒运动）、Zb052（潮汐与地月）、Zb086（偏置量）。
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "index.src.html")
OUT = os.path.join(HERE, "index.html")
THREE = os.path.join(HERE, "three.min.js")

CHECKS = []


def check(name, got, want, tol, unit="", rel=False):
    err = abs(got - want)
    if rel:
        err = err / abs(want) if want else err
    ok = err <= tol
    CHECKS.append((ok, name, got, want, err, unit, rel))
    return ok


# ---------------------------------------------------------------- 行星表
# 名称, 半长轴 AU, 周期 yr, 偏心率, 倾角°, 近日点经度°, J2000 平近点角°,
# 颜色, 显示半径（AU；已按可见性放缩）
PLANETS = [
    ("水星", 0.38710, 0.240846, 0.2056, 7.005, 77.456, 174.796, "#b9a48a", 0.085),
    ("金星", 0.72333, 0.615198, 0.0068, 3.395, 131.532, 50.115, "#e8c37a", 0.130),
    ("地球", 1.00000, 1.000017, 0.0167, 0.000, 102.937, 357.517, "#6fa8dc", 0.135),
    ("火星", 1.52368, 1.880848, 0.0934, 1.850, 336.060, 19.373, "#d98b6a", 0.100),
    ("木星", 5.20260, 11.8620, 0.0484, 1.303, 14.753, 20.020, "#d9b08c", 0.380),
    ("土星", 9.55491, 29.4570, 0.0542, 2.485, 92.432, 317.020, "#e6d3a3", 0.320),
    ("天王星", 19.21845, 84.0205, 0.0472, 0.773, 170.964, 142.238, "#9fd8dc", 0.220),
    ("海王星", 30.11039, 164.790, 0.0086, 1.770, 44.971, 256.228, "#7f9fe0", 0.210),
    ("冥王星", 39.4820, 247.940, 0.2488, 17.16, 224.07, 14.86, "#b0a8c0", 0.060),
]

# 壳层判据用的十个"天体"（含小行星带，Zb050 §4 的表）
SHELL_A = [0.38710, 0.72333, 1.00000, 1.52368, 2.70000,
           5.20288, 9.53667, 19.18916, 30.06992, 39.48200]
SHELL_N = ["水", "金", "地", "火", "带", "木", "土", "天", "海", "冥"]


def shell_stats(a_list, r0):
    n = [math.log2(a / r0) for a in a_list]
    res = [abs(x - round(x)) for x in n]
    return n, res


def f_metric(res):
    return math.sqrt(sum(r * r for r in res) / len(res))


# ---------------------------------------------------------------- 格点力
def lattice_force(N=96):
    """三维周期格上的 G=L^+（零模投影），返回 |F|4πr² 表与 r=10 环上的方向依赖。"""
    k = 2 * np.pi * np.fft.fftfreq(N)
    KX, KY, KZ = np.meshgrid(k, k, k, indexing="ij")
    lam = 4 * (np.sin(KX / 2) ** 2 + np.sin(KY / 2) ** 2 + np.sin(KZ / 2) ** 2)
    lam[0, 0, 0] = 1.0
    Gk = 1.0 / lam
    Gk[0, 0, 0] = 0.0
    G = np.fft.ifftn(Gk).real

    def g(x, y, z):
        return G[x % N, y % N, z % N]

    def force(x, y, z, h=1):
        return np.array([
            -(g(x + h, y, z) - g(x - h, y, z)) / (2 * h),
            -(g(x, y + h, z) - g(x, y - h, z)) / (2 * h),
            -(g(x, y, z + h) - g(x, y, z - h)) / (2 * h),
        ])

    rows = []
    for r in (6, 8, 10, 12, 16):
        mag = float(np.linalg.norm(force(r, 0, 0)))
        rows.append(dict(r=r, f=mag * 4 * math.pi * r * r))

    aniso = []
    for (x, y) in ((10, 0), (8, 6), (7, 7)):
        r = math.hypot(x, y)
        mag = float(np.linalg.norm(force(x, y, 0)))
        aniso.append(dict(r=r, th=math.degrees(math.atan2(y, x)),
                          f=mag * 4 * math.pi * r * r))
    return rows, aniso


# ---------------------------------------------------------------- 地月
def earth_moon():
    G = 6.67430e-11
    ME, MM = 5.9722e24, 7.342e22
    RE, RM = 6.371e6, 1.7374e6
    a0 = 384.4e6
    IE = 0.3307 * ME * RE ** 2          # 地球转动惯量（0.3307 M R^2）
    IM = 0.3929 * MM * RM ** 2          # 月球（均匀球近似）
    mu = ME * MM / (ME + MM)
    GM = G * (ME + MM)
    wE = 2 * math.pi / 86164.1          # 恒星日
    TM = 27.321661 * 86400.0
    wM = 2 * math.pi / TM

    L_orb = mu * math.sqrt(GM * a0)
    L_spin = IE * wE + IM * wM
    L_tot = L_orb + L_spin

    # 终态：双同步，F(a)=L_tot
    lo, hi = a0 * 1.0001, 2.0e9
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        F = mu * math.sqrt(GM * mid) + (IE + IM) * math.sqrt(GM / mid ** 3)
        if F < L_tot:
            lo = mid
        else:
            hi = mid
    af = 0.5 * (lo + hi)
    Tf = 2 * math.pi * math.sqrt(af ** 3 / GM)
    a_sync = (GM / wE ** 2) ** (1 / 3)
    a_star = math.sqrt(3 * (IE + IM) / mu)

    # 自洽性：由 L 守恒反推当前日长
    day0 = 2 * math.pi * IE / (L_tot - mu * math.sqrt(GM * a0))
    return dict(a0=a0 / 1e3, af=af / 1e3, Tf=Tf / 86400, a_sync=a_sync / 1e3,
                a_star=a_star / 1e3, GM=GM, mu=mu, IE=IE, IM=IM, Ltot=L_tot,
                day0=day0 / 3600.0, Tmonth0=27.321661,
                LorbFrac0=L_orb / L_tot)


# ---------------------------------------------------------------- 太阳偏置
def solar_bias():
    AU = 1.495978707e8  # km
    MJ = 1047.3486      # M_sun / M_jup
    rows = []
    data = [("木星", 5.20288, 1.000000), ("土星", 9.53667, 0.299400),
            ("海王星", 30.06992, 0.053720), ("天王星", 19.18916, 0.045720),
            ("地球", 1.00000, 0.003147)]
    for name, a, m in data:
        b = a * AU * m / (MJ + m)  # km
        rows.append(dict(name=name, a=a, m=m, b=b))
    return rows


def main():
    R = {}

    R["planets"] = [dict(name=n, a=a, T=T, e=e, inc=i, w=w, m0=m0, color=c, rv=rv)
                    for n, a, T, e, i, w, m0, c, rv in PLANETS]

    # T^2 ∝ a^3：以 4π²a³/T² 归一
    ref = 4 * math.pi ** 2 / PLANETS[2][2] ** 2
    R["kepler_dev"] = [dict(name=n, a=a, T=T,
                            dev=(4 * math.pi ** 2 * a ** 3 / T ** 2 / ref - 1) * 100)
                       for n, a, T, _, _, _, _, _, _ in PLANETS]

    # 壳层
    r0_cal = (5.20288 / 16 * 9.53667 / 32 * 19.18916 / 64) ** (1 / 3)
    n_cal, res_cal = shell_stats(SHELL_A, r0_cal)
    F_all = f_metric(res_cal)
    keep = [r for i, r in enumerate(res_cal) if i not in (5, 6, 7)]
    F_keep = f_metric(keep)

    r0_pred = 0.3109  # Zb050 (4.5)：由不含太阳系量的常数组合给出
    n_pred, res_pred = shell_stats(SHELL_A, r0_pred)
    R["shell"] = dict(names=SHELL_N, a=SHELL_A, r0_cal=r0_cal, r0_pred=r0_pred,
                      n=n_cal, res=res_cal, res_pred=res_pred,
                      F_all=F_all, F_keep=F_keep, calib=[5, 6, 7])

    rows, aniso = lattice_force()
    R["lattice"] = dict(rows=rows, aniso=aniso)

    em = earth_moon()
    R["em"] = em

    bias = solar_bias()
    R["bias"] = bias
    R["bias_sum"] = sum(r["b"] for r in bias[:4])
    R["Rsun"] = 6.957e5  # km

    R["alpha"] = 137.035999084
    R["ratio5a"] = 5 / 137.035999084
    R["ratio_obs"] = 0.99726968 / 27.321661

    # ------------------------------------------------------------ 核验
    print("=" * 66)
    print("核验（与篇目登记值对照）")
    print("=" * 66)

    def line(name, got, want, tol, unit="", rel=False):
        ok = check(name, got, want, tol, unit, rel)
        print(f"{'PASS' if ok else 'FAIL'}  {name:28s} 算得 {got:14.6g}  登记 {want:14.6g} {unit}")

    line("壳层 F_all", F_all, 0.2300, 2e-3)
    line("壳层 F_keep", F_keep, 0.2723, 2e-3)
    line("r0 定标", r0_cal, 0.3074, 5e-4)
    line("地月 a_f/a0", em["af"] / em["a0"], 1.44, 0.01)
    line("地月 T_f (d)", em["Tf"], 47.2, 0.1)
    line("地月 a_sync (km)", em["a_sync"], 42336, 200)
    line("地月 a* (km)", em["a_star"], 57600, 300)
    line("地月当前日长 (h)", em["day0"], 23.934, 5e-3)
    line("5/α⁻¹ vs 恒星日/恒星月", (R["ratio5a"] / R["ratio_obs"] - 1) * 100, -0.039, 5e-3, "%")
    line("木星偏置 (km)", bias[0]["b"], 742444, 200)
    line("木星偏置/R⊙", bias[0]["b"] / R["Rsun"], 1.067, 1e-3)
    line("格点 |F|4πr² (r=6)", rows[0]["f"], 1.055, 2e-3)
    line("格点 |F|4πr² (r=16)", rows[-1]["f"], 0.986, 2e-3)
    line("格点 r=10 轴 +", [a for a in aniso if abs(a["th"]) < 1e-9][0]["f"], 1.0134, 1e-3)
    line("格点 r=10 对角 −", [a for a in aniso if abs(a["th"] - 45) < 1e-9][0]["f"], 0.9914, 2e-3)

    print("-" * 66)
    print(f"壳层：r0={r0_cal:.6f} AU（木/土/天 定标）；F_all={F_all:.4f}，F_keep={F_keep:.4f}")
    print(f"地月：a0={em['a0']:.0f} km → a_f={em['af']:.0f} km（×{em['af']/em['a0']:.3f}），"
          f"T_f={em['Tf']:.2f} d，同步半径 {em['a_sync']:.0f} km")
    print(f"格点：{[(r['r'], round(r['f'],4)) for r in rows]}")
    print(f"偏置：木星 {bias[0]['b']:.0f} km = {bias[0]['b']/R['Rsun']:.3f} R⊙；"
          f"四巨行星同向合计 {R['bias_sum']:.0f} km = {R['bias_sum']/R['Rsun']:.2f} R⊙")

    nfail = sum(1 for c in CHECKS if not c[0])
    print("-" * 66)
    print(f"核验：{len(CHECKS)-nfail}/{len(CHECKS)} PASS")

    # ------------------------------------------------------------ 构建
    with open(SRC, encoding="utf-8") as f:
        html = f.read()
    blob = json.dumps(R, ensure_ascii=False, separators=(",", ":"))
    js = "/*DATA-BEGIN*/\nconst DATA = " + blob + ";\n/*DATA-END*/"
    for begin, end, payload in (
        ("/*DATA-BEGIN*/", "/*DATA-END*/", js),
        ("/*THREE-BEGIN*/", "/*THREE-END*/", THREESRC),
    ):
        i, j = html.find(begin), html.find(end)
        if i < 0 or j < 0:
            print(f"!! 源文件里找不到标记 {begin}，跳过构建")
            return 1 if nfail else 0
        html = html[:i] + payload + html[j + len(end):]
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"已构建 {OUT}（{len(html)/1024:.0f} KB，含内联 three.js 与数据）")
    return 1 if nfail else 0


THREESRC = open(THREE, encoding="utf-8").read() if os.path.exists(THREE) else ""

if __name__ == "__main__":
    sys.exit(main())
