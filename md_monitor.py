#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VASP AIMD 生产阶段监控脚本
  - 温度：平均值、涨落（与正则系综理论值对比）、分块平均、趋势
  - 能量：势能 F / E0 分块平均与趋势；守恒量 E 的漂移（Nosé / NVE 时有意义）
  - 磁矩、每个离子步的电子步数、SCF 未收敛的步
  - 计时：平均单步耗时、预计剩余时间（读 OUTCAR 的 LOOP+）
  - 输出 md_data.csv；若装有 matplotlib 再输出 md_monitor.png

只依赖 Python 标准库（画图需要 matplotlib，没有则自动跳过）。
在计算目录下运行，读取 OSZICAR、INCAR、POSCAR、OUTCAR（后三者缺失也能运行）。

用法:
    python3 md_monitor.py                 # 统计全部步数
    python3 md_monitor.py --skip 2000     # 丢掉前 2000 步（平衡段）再统计
    python3 md_monitor.py --block 400     # 分块平均的块大小（步），默认 200
    python3 md_monitor.py --dir ../prod   # 指定计算目录
    python3 md_monitor.py --no-plot       # 不画图
"""
import argparse
import math
import os
import re
import sys
import io

if sys.version_info < (3, 6):
    sys.exit("需要 Python >= 3.6，请用 python3 运行")
# Python 3.6 在 LANG=C 等非 UTF-8 环境下打印中文会报 UnicodeEncodeError
if (getattr(sys.stdout, "encoding", None) or "").lower().replace("-", "") != "utf8":
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
    except Exception:
        pass

# CSV 表头中每列的含义（与 OSZICAR 中离子步那一行的字段一一对应）
CSV_LABEL = {
    "step":  "离子步序号",
    "T":     "温度(K)",
    "E":     "总能量(eV)=F+EK+SP+SK；Nosé/NVE 时为守恒量",
    "F":     "自由能(eV)；即势能，原子受力由它求导而来",
    "E0":    "σ→0 外推能量(eV)",
    "EK":    "离子动能(eV)",
    "SP":    "Nosé 热浴势能(eV)",
    "SK":    "Nosé 热浴动能(eV)",
    "mag":   "总磁矩(μB)",
    "nelec": "该步电子步数",
}

MDALGO_NAME = {0: "标准 MD（SMASS 决定恒温方式）", 1: "Andersen", 2: "Nosé-Hoover",
               3: "Langevin", 4: "Nosé-Hoover chain", 5: "CSVR", 13: "Multiple Andersen"}


# ---------------------------------------------------------------- 读取文件
def read_incar(path):
    tags = {}
    if not os.path.exists(path):
        return tags
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = re.split(r"[#!]", line)[0]
            for part in line.split(";"):
                if "=" in part:
                    k, v = part.split("=", 1)
                    tags[k.strip().upper()] = v.strip()
    return tags


def tag_num(tags, key, default, typ=float):
    try:
        return typ(float(tags[key].split()[0]))
    except (KeyError, ValueError, IndexError):
        return default


def read_natoms(d):
    for fn in ("POSCAR", "CONTCAR"):
        p = os.path.join(d, fn)
        if os.path.exists(p):
            with open(p, encoding="utf-8", errors="ignore") as f:
                lines = [f.readline() for _ in range(7)]
            for ln in (lines[6], lines[5]):
                try:
                    n = sum(int(x) for x in ln.split())
                    if n > 0:
                        return n
                except ValueError:
                    continue
    return None


ION_RE = re.compile(r"^\s*(\d+)\s+T=")
KV_RE = re.compile(r"([A-Za-z0-9]+)=\s*(-?[0-9.]+(?:[Ee][+-]?\d+)?)")
ELEC_RE = re.compile(r"^\s*(DAV|RMM|CG|SDA)\s*:")


def read_oszicar(path):
    steps, nel = [], 0
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            if ELEC_RE.match(line):
                nel += 1
                continue
            m = ION_RE.match(line)
            if m:
                d = {k: float(v) for k, v in KV_RE.findall(line)}
                d["step"] = int(m.group(1))
                d["nelec"] = nel
                nel = 0
                steps.append(d)
    return steps


def read_loop_times(path):
    t = []
    if not os.path.exists(path):
        return t
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "LOOP+" in line:
                try:
                    t.append(float(line.split()[-1]))
                except ValueError:
                    pass
    return t


# ---------------------------------------------------------------- 统计工具
def mean(a):
    return sum(a) / len(a) if a else float("nan")


def std(a):
    if len(a) < 2:
        return float("nan")
    m = mean(a)
    return math.sqrt(sum((x - m) ** 2 for x in a) / len(a))


def slope(xs, ys):
    n = len(xs)
    if n < 2:
        return float("nan")
    mx, my = mean(xs), mean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return float("nan")
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx


def running(a, w):
    out, s = [], 0.0
    for i, x in enumerate(a):
        s += x
        if i >= w:
            s -= a[i - w]
        out.append(s / min(i + 1, w))
    return out


# ---------------------------------------------------------------- 主程序
def main():
    ap = argparse.ArgumentParser(description="VASP AIMD 温度/能量监控")
    ap.add_argument("--dir", default=".", help="计算目录（默认当前目录）")
    ap.add_argument("--skip", type=int, default=0, help="统计时丢弃的前 N 步（平衡段）")
    ap.add_argument("--block", type=int, default=200, help="分块平均的块大小（步）")
    ap.add_argument("--short-header", action="store_true",
                    help="CSV 只用简短英文表头（便于 pandas/gnuplot 读取）")
    ap.add_argument("--no-plot", action="store_true", help="不画图")
    args = ap.parse_args()

    d = args.dir
    osz = os.path.join(d, "OSZICAR")
    if not os.path.exists(osz):
        sys.exit(f"找不到 {osz}")

    tags = read_incar(os.path.join(d, "INCAR"))
    potim = tag_num(tags, "POTIM", 0.5)
    nsw = tag_num(tags, "NSW", 0, int)
    nelm = tag_num(tags, "NELM", 60, int)
    t0 = tag_num(tags, "TEBEG", float("nan"))
    mdalgo = tag_num(tags, "MDALGO", 0, int)
    smass = tag_num(tags, "SMASS", 0.0)
    natoms = read_natoms(d)

    data = read_oszicar(osz)
    if not data:
        sys.exit("OSZICAR 中还没有完成的离子步")

    nstep = len(data)
    sel = [s for s in data if s["step"] > args.skip]
    if not sel:
        sys.exit(f"已完成 {nstep} 步，全部在 --skip {args.skip} 范围内，暂无可统计的数据")

    def col(rows, k):
        return [r.get(k, float("nan")) for r in rows]

    T, F, E0, E = col(sel, "T"), col(sel, "F"), col(sel, "E0"), col(sel, "E")
    EK, MAG, NEL = col(sel, "EK"), col(sel, "mag"), col(sel, "nelec")
    tps = [r["step"] * potim / 1000.0 for r in sel]          # 时间 (ps)

    # ---------------- 基本信息
    print("=" * 64)
    print(f"VASP AIMD 监控   目录: {os.path.abspath(d)}")
    print("=" * 64)
    prog = f" / {nsw} 步 ({100.0 * nstep / nsw:.1f}%)" if nsw else " 步"
    print(f"已完成 {nstep}{prog}，POTIM = {potim} fs，已模拟 {nstep * potim / 1000:.3f} ps")
    print(f"恒温器: MDALGO = {mdalgo} ({MDALGO_NAME.get(mdalgo, '未知')})，"
          f"目标温度 TEBEG = {t0:g} K，原子数 = {natoms if natoms else '未知'}")
    print(f"统计范围: 第 {sel[0]['step']} - {sel[-1]['step']} 步"
          f"（共 {len(sel)} 步，{len(sel) * potim / 1000:.3f} ps"
          f"{'，已丢弃前 %d 步' % args.skip if args.skip else ''}）")

    # ---------------- 温度
    print("\n--- 温度 ---")
    sT = std(T)
    line = f"平均 T = {mean(T):.1f} K，标准差 = {sT:.1f} K"
    sig_th = None
    if natoms and not math.isnan(t0):
        sig_th = t0 * math.sqrt(2.0 / (3 * natoms))
        line += f"（正则系综理论涨落约 {sig_th:.1f} K）"
    print(line)
    print(f"最低 / 最高 T = {min(T):.0f} / {max(T):.0f} K")
    if len(T) >= 4:
        h = len(T) // 2
        print(f"前半段 / 后半段平均 T = {mean(T[:h]):.1f} / {mean(T[h:]):.1f} K，"
              f"线性趋势 {slope(tps, T):+.1f} K/ps")
    if sig_th:
        out = [r["step"] for r, t in zip(sel, T) if abs(t - t0) > 4 * sig_th]
        if out:
            print(f"[注意] {len(out)} 步偏离目标温度超过 4 倍理论涨落 (±{4 * sig_th:.0f} K)，"
                  f"例如第 {', '.join(map(str, out[:8]))}{' ...' if len(out) > 8 else ''} 步")

    # ---------------- 能量
    print("\n--- 能量 (eV) ---")
    print(f"势能 F : 平均 {mean(F):.4f}，标准差 {std(F):.4f}，线性趋势 {slope(tps, F):+.4f} eV/ps")
    print(f"E0     : 平均 {mean(E0):.4f}，标准差 {std(E0):.4f}")
    print(f"动能 EK: 平均 {mean(EK):.4f}")
    conserved = (mdalgo == 2) or (mdalgo == 0 and (smass >= 0 or smass == -3))
    if conserved:
        dr = slope(tps, E)
        txt = f"守恒量 E（含恒温器能量）: 漂移 {dr:+.4f} eV/ps"
        if natoms:
            txt += f" = {1000 * dr / natoms:+.3f} meV/atom/ps"
        txt += f"，全程变化范围 {max(E) - min(E):.4f} eV"
        print(txt)
        print("        （经验参考：|漂移| 在 1 meV/atom/ps 量级以内通常可以接受；"
              "明显更大说明步长偏大或 SCF 精度不足）")
    else:
        print(f"守恒量: MDALGO = {mdalgo} 时 OSZICAR 中的 E 不是守恒量，不做漂移检查")

    # ---------------- 磁矩与 SCF
    if not all(math.isnan(m) for m in MAG):
        print(f"\n--- 磁矩 ---\n平均 {mean(MAG):.3f}，范围 {min(MAG):.3f} ~ {max(MAG):.3f} μB")
    print(f"\n--- SCF ---\n平均每个离子步 {mean(NEL):.1f} 个电子步（最多 {max(NEL)}，NELM = {nelm}）")
    bad = [r["step"] for r in sel if r["nelec"] >= nelm]
    if bad:
        print(f"[注意] {len(bad)} 步达到 NELM 上限，SCF 可能未收敛："
              f"第 {', '.join(map(str, bad[:10]))}{' ...' if len(bad) > 10 else ''} 步")
    else:
        print("所有统计步的 SCF 都在 NELM 之内收敛")

    # ---------------- 分块平均
    b = max(1, args.block)
    print(f"\n--- 分块平均（每块 {b} 步 = {b * potim / 1000:.3f} ps）---")
    print(f"{'步数范围':>15s} {'<T> (K)':>9s} {'<F> (eV)':>13s} {'<E0> (eV)':>13s} "
          f"{'<mag>':>7s} {'<电子步>':>8s}")
    for i in range(0, len(sel), b):
        ch = sel[i:i + b]
        mg = mean(col(ch, "mag"))
        print(f"{ch[0]['step']:>7d}-{ch[-1]['step']:<7d} {mean(col(ch, 'T')):9.1f} "
              f"{mean(col(ch, 'F')):13.4f} {mean(col(ch, 'E0')):13.4f} "
              f"{mg:7.3f} {mean(col(ch, 'nelec')):8.1f}"
              f"{'   (不足一块)' if len(ch) < b else ''}")

    # ---------------- 最新状态
    print("\n--- 最近 5 步 ---")
    for r in data[-5:]:
        print(f"  第 {r['step']:6d} 步  T = {r.get('T', float('nan')):7.1f} K  "
              f"F = {r.get('F', float('nan')):.4f} eV  mag = {r.get('mag', float('nan')):.3f}  "
              f"电子步 {r['nelec']}")

    # ---------------- 计时
    lt = read_loop_times(os.path.join(d, "OUTCAR"))
    if len(lt) >= 2:
        avg = mean(lt[1:])
        print(f"\n--- 计时（OUTCAR LOOP+，去掉第 1 步）---\n平均每步 {avg:.1f} s")
        if nsw and nsw > nstep:
            rem = (nsw - nstep) * avg / 3600.0
            print(f"剩余 {nsw - nstep} 步，预计还需 {rem:.1f} h（约 {rem / 24:.1f} 天）")

    # ---------------- 输出 CSV
    csv = os.path.join(d, "md_data.csv")
    keys = ["step", "T", "E", "F", "E0", "EK", "SP", "SK", "mag", "nelec"]
    if args.short_header:
        header = ["time_ps"] + keys
    else:
        header = ["time_ps 模拟时间(ps)"] + [k + " " + CSV_LABEL[k] for k in keys]
    # utf-8-sig 带 BOM，Windows 下用 Excel 直接打开中文不乱码
    with open(csv, "w", encoding="utf-8-sig") as f:
        f.write(",".join(header) + "\n")
        for r in data:
            f.write(f"{r['step'] * potim / 1000:.4f}," +
                    ",".join(str(r.get(k, "")) for k in keys) + "\n")
    print(f"\n逐步数据已写入 {csv}")

    # ---------------- 画图
    if args.no_plot:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("未安装 matplotlib 或无法加载，跳过画图（可把 md_data.csv 拷到本地用 Excel/Origin 作图）")
        return

    x = [r["step"] * potim / 1000.0 for r in data]
    w = min(b, len(data))
    aT, aF, aE0, aE = col(data, "T"), col(data, "F"), col(data, "E0"), col(data, "E")
    aM, aN = col(data, "mag"), col(data, "nelec")

    fig, ax = plt.subplots(5, 1, figsize=(9, 13), sharex=True)
    ax[0].plot(x, aT, lw=0.5, color="0.6", label="T")
    ax[0].plot(x, running(aT, w), color="C3", label=f"running avg ({w} steps)")
    if not math.isnan(t0):
        ax[0].axhline(t0, color="k", ls="--", lw=0.8, label=f"target {t0:g} K")
        if sig_th:
            ax[0].axhspan(t0 - sig_th, t0 + sig_th, color="C2", alpha=0.15,
                          label="±1σ canonical")
    ax[0].set_ylabel("T (K)")
    ax[0].legend(fontsize=8, loc="upper right")

    ax[1].plot(x, aF, lw=0.5, color="C0", label="F")
    ax[1].plot(x, running(aF, w), color="C1", label="F running avg")
    ax[1].set_ylabel("F (eV)")
    ax[1].legend(fontsize=8, loc="upper right")

    ax[2].plot(x, aE, lw=0.8, color="C4")
    ax[2].set_ylabel("E (eV)\n" + ("conserved" if conserved else "not conserved"))

    ax[3].plot(x, aM, lw=0.8, color="C5")
    ax[3].set_ylabel("mag (μB)")

    ax[4].bar(x, aN, width=potim / 1000.0, color="C7")
    ax[4].axhline(nelm, color="r", ls="--", lw=0.8, label=f"NELM = {nelm}")
    ax[4].set_ylabel("electronic steps")
    ax[4].set_xlabel("time (ps)")
    ax[4].legend(fontsize=8, loc="upper right")

    if args.skip > 0:
        for a in ax:
            a.axvspan(0, args.skip * potim / 1000.0, color="0.85", alpha=0.5)
        ax[0].text(0.01, 0.95, "discarded (equilibration)", transform=ax[0].transAxes,
                   fontsize=8, va="top")

    png = os.path.join(d, "md_monitor.png")
    try:
        fig.tight_layout()
        fig.savefig(png, dpi=130)
        print(f"图已保存到 {png}")
    except Exception as e:
        print(f"画图失败（{e}），可用 md_data.csv 自行作图")


if __name__ == "__main__":
    main()
