"""
Diffusion Coefficient Calculation from Z-trajectory Files
==========================================================
Pipeline:
  1. Read z1_{i}.dat files (columns: time, z-position)
  2. Compute fluctuations: dz = z - mean(z)
  3. Compute autocorrelation C(t) via FFT (zero-padded, unbiased)
  4. Find tc = first negative zero-crossing of C(t)
  5. Integrate C(t) numerically from 0 → tc  (tau_0_to_tc, raw Å²·lag)
  6. Fit C(t) = A·exp(-γt)·cos(ωt) to the tail (tc → end)
  7. Integrate fit analytically from tc → ∞ (tau_fit, raw Å²·lag)
  8. Normalize both taus by variance:
       tau_total_norm = (tau_0_to_tc + tau_fit) / var
  9. D = var / tau_total_norm   [= var² / (tau_0_to_tc + tau_fit)]
 10. Save CSV, TXT, and plots
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend (safe for servers)
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit
from scipy.signal import argrelextrema


DATA_DIR    = "/data/Kunal/3_glycerol/remd/analysis2.0/window_z_traj/"
FILE_RANGE  = range(-30, 31)          # indices for z1_{i}.dat
OUTPUT_DIR  = "."                      # where to save CSV, TXT, PNGs
SKIPROWS    = 1                        # header lines to skip in each .dat



def read_data(data_dir, file_range, skiprows=1):
    """Read all z trajectory files. Returns {index: ndarray(N,2)}."""
    data_dict = {}
    for i in file_range:
        path = os.path.join(data_dir, f"z1_{i}.dat")
        data = np.loadtxt(path, skiprows=skiprows)
        data_dict[i] = data
        print(f"  Read {path}: {len(data)} rows")
    print(f"\nLoaded {len(data_dict)} files.\n")
    return data_dict


def compute_dz(data_dict):
    """Compute mean-subtracted z fluctuations: dz = z - mean(z)."""
    dz_dict = {}
    for i, data in data_dict.items():
        z = data[:, 1]
        dz_dict[i] = z - np.mean(z)
    return dz_dict


def compute_fft_acf(dz_dict):
    """
    Compute unbiased autocorrelation C(m) via FFT (zero-padded).
    C(m) = IFFT(|FFT(dz_padded)|²)[m] / (N - m)
    Units: Å² (same as variance of dz).
    """
    C_fft_dict = {}
    for i, dz in dz_dict.items():
        N = len(dz)
        dz_padded = np.concatenate([dz, np.zeros(N - 1)])
        F_dz      = np.fft.fft(dz_padded)
        IFT       = np.fft.ifft(np.abs(F_dz) ** 2).real
        C_fft     = np.array([IFT[m] / (N - m) for m in range(N)])
        C_fft_dict[i] = C_fft
    return C_fft_dict


def find_tc_and_tau(C_fft_dict):
    """
    For each file:
      tc  = index of first negative value in C(t)
      tau = trapz integral of C(t) from 0 to tc   [Å²·lag, raw, unnormalised]
    """
    tc_tau_dict = {}
    for i, C in C_fft_dict.items():
        neg_idx = np.where(C < 0)[0]
        tc = neg_idx[0] if len(neg_idx) > 0 else len(C) - 1
        m_vals = np.arange(tc + 1)
        tau    = np.trapz(C[:tc + 1], m_vals)      # Å²·lag
        tc_tau_dict[i] = {'tc': tc, 'tau': tau}
        print(f"  z1_{i:+d}.dat  tc={tc:4d}  tau(0→tc)={tau:.4f} Å²·lag")
    return tc_tau_dict


# ----- Damped-cosine fitting  -----

def damped_cosine(t, A, gamma, omega):
    return A * np.exp(-gamma * t) * np.cos(omega * t)


def _estimate_gamma(t_tail, C_tail):
    """Estimate decay rate from consecutive peak ratios."""
    pos_idx = argrelextrema(C_tail, np.greater)[0]
    neg_idx = argrelextrema(C_tail, np.less)[0]
    estimates = []
    for idx_arr in (pos_idx, neg_idx):
        if len(idx_arr) < 2:
            continue
        for k in range(min(len(idx_arr) - 1, 5)):
            i1, i2 = idx_arr[k], idx_arr[k + 1]
            P1, P2 = abs(C_tail[i1]), abs(C_tail[i2])
            dt = t_tail[i2] - t_tail[i1]
            if P1 > P2 > 0 and dt > 0:
                g = np.log(P1 / P2) / dt
                if 0 < g < 1:
                    estimates.append(g)
    return np.median(estimates) if estimates else 1e-3


def _estimate_omega(t_tail, C_tail):
    """Estimate oscillation frequency from peak spacing."""
    pos_idx = argrelextrema(C_tail, np.greater)[0]
    neg_idx = argrelextrema(C_tail, np.less)[0]
    periods = []
    for idx_arr in (pos_idx, neg_idx):
        if len(idx_arr) < 2:
            continue
        for k in range(min(len(idx_arr) - 1, 5)):
            T = t_tail[idx_arr[k + 1]] - t_tail[idx_arr[k]]
            if T > 0:
                periods.append(T)
    if periods:
        return 2 * np.pi / np.median(periods)
    # Fallback: zero-crossing
    zc = np.where(np.diff(np.sign(C_tail)))[0]
    if len(zc) > 2:
        return np.pi / np.median(np.diff(zc))
    return 0.01


def fit_tail(C_fft_dict, tc_tau_dict, dz_dict):
    """
    Fit damped cosine to the tail (tc → end) of each C(t).
    Analytical integral from tc → ∞:
      tau_fit_raw = A·exp(-γ·tc)·γ / (γ² + ω²)   [Å²·lag]
    Returns fit_params_dict with 'tau_fit_raw' key.
    """
    fit_params_dict = {}

    for i in sorted(C_fft_dict.keys()):
        C   = C_fft_dict[i]
        tc  = tc_tau_dict[i]['tc']
        var = np.var(dz_dict[i])

        t_tail = np.arange(tc, len(C), dtype=float)
        C_tail = C[tc:]

        A_init     = C[0]                            # C(0) as amplitude seed
        gamma_init = _estimate_gamma(t_tail, C_tail)
        omega_init = _estimate_omega(t_tail, C_tail)

        try:
            popt, pcov = curve_fit(
                damped_cosine,
                t_tail,
                C_tail,
                p0=[A_init, gamma_init, omega_init],
                bounds=([0, 0, 0], [max(10 * A_init, 1e-6), 1.0, 10 * omega_init]),
                maxfev=20000,
            )
            A_fit, gamma_fit, omega_fit = popt
            perr = np.sqrt(np.diag(pcov))

            # Analytical integral of A·exp(-γt)·cos(ωt) from tc to ∞
            #   = A·exp(-γ·tc)·γ / (γ² + ω²)      [exact for γ > 0]
            if gamma_fit > 0:
                tau_fit_raw = (A_fit * np.exp(-gamma_fit * tc) * gamma_fit /
                               (gamma_fit ** 2 + omega_fit ** 2))
            else:
                tau_fit_raw = np.nan

            C_fit    = damped_cosine(t_tail, *popt)
            ss_res   = np.sum((C_tail - C_fit) ** 2)
            ss_tot   = np.sum((C_tail - np.mean(C_tail)) ** 2)
            r2       = 1 - ss_res / ss_tot if ss_tot != 0 else 0.0

            fit_params_dict[i] = {
                'tc': tc, 'A': A_fit, 'gamma': gamma_fit, 'omega': omega_fit,
                'A_err': perr[0], 'gamma_err': perr[1], 'omega_err': perr[2],
                'r_squared': r2, 'tau_fit_raw': tau_fit_raw,
            }
            print(f"  z1_{i:+d}.dat  A={A_fit:.4f}  γ={gamma_fit:.5f}  "
                  f"ω={omega_fit:.5f}  R²={r2:.4f}  tau_fit_raw={tau_fit_raw:.4f}")

        except Exception as exc:
            print(f"  z1_{i:+d}.dat  FIT FAILED: {exc}")
            fit_params_dict[i] = {
                'tc': tc, 'A': np.nan, 'gamma': np.nan, 'omega': np.nan,
                'A_err': np.nan, 'gamma_err': np.nan, 'omega_err': np.nan,
                'r_squared': np.nan, 'tau_fit_raw': np.nan,
            }

    return fit_params_dict


def compute_diffusion(C_fft_dict, tc_tau_dict, fit_params_dict, dz_dict):
    """
    Compute diffusion coefficient for each window.

    tau_0_to_tc  [Å²·lag] — raw trapz integral 0 → tc
    tau_fit_raw  [Å²·lag] — raw analytical integral tc → ∞
    tau_total    [Å²·lag] = tau_0_to_tc + tau_fit_raw
    tau_total_norm [lag]  = tau_total / var

    D = var / tau_total_norm = var² / tau_total
    """
    results = {}
    print(f"\n{'File':<12} {'z':>6} {'var':>10} {'tau(0→tc)':>12} "
          f"{'tau_fit':>12} {'tau_total':>12} {'D':>12}")
    print("-" * 80)

    for i in sorted(C_fft_dict.keys()):
        var          = np.var(dz_dict[i])
        tau_0_to_tc  = tc_tau_dict[i]['tau']          # Å²·lag  (raw)
        tau_fit_raw  = fit_params_dict[i]['tau_fit_raw']  # Å²·lag  (raw)
        C_0          = C_fft_dict[i][0]

        tau_total = tau_0_to_tc + tau_fit_raw         # Å²·lag

        if tau_total > 0 and not np.isnan(tau_total):
            # D = var² / tau_total  =  var / tau_total_norm
            D = (var ** 2) / tau_total
        else:
            D = np.nan

        results[i] = {
            'z':           i,
            'var':         var,
            'C_0':         C_0,
            'tau_0_to_tc': tau_0_to_tc,
            'tau_fit_raw': tau_fit_raw,
            'tau_total':   tau_total,
            'D':           D,
        }
        print(f"  z1_{i:+d}.dat  {i:>6}  {var:>10.4f}  "
              f"{tau_0_to_tc:>12.4f}  {tau_fit_raw:>12.4f}  "
              f"{tau_total:>12.4f}  {D:>12.4f}")

    return results


def save_results(results, output_dir):
    """Save results to CSV and a formatted TXT file."""
    df = pd.DataFrame.from_dict(results, orient='index')
    df.index.name = 'file_index'
    df = df.reset_index().sort_values('z')

    csv_path = os.path.join(output_dir, "diffusion_coefficients.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Results saved to: {csv_path}")

    txt_path = os.path.join(output_dir, "diffusion_coefficients.txt")
    with open(txt_path, 'w') as f:
        f.write("Diffusion Coefficient Analysis\n")
        f.write("=" * 90 + "\n\n")
        f.write("Method:\n")
        f.write("  C(t) computed via zero-padded FFT (unbiased)\n")
        f.write("  tc   = first negative zero-crossing of C(t)\n")
        f.write("  tau(0→tc)   = trapz integral of C(t) from 0 to tc       [Å²·lag]\n")
        f.write("  tau_fit_raw = analytical integral of fitted damped cosine  [Å²·lag]\n")
        f.write("                fit: A·exp(-γt)·cos(ωt), integral = A·exp(-γ·tc)·γ/(γ²+ω²)\n")
        f.write("  tau_total   = tau(0→tc) + tau_fit_raw                     [Å²·lag]\n")
        f.write("  D           = var² / tau_total                             [Å²/lag]\n\n")
        f.write("=" * 90 + "\n")
        header = (f"{'File':<10} {'z':>6} {'var':>10} {'tau(0→tc)':>12} "
                  f"{'tau_fit':>12} {'tau_total':>12} {'D':>12}\n")
        f.write(header)
        f.write("=" * 90 + "\n")
        for _, row in df.iterrows():
            f.write(
                f"z1_{int(row['z']):+d}.dat  {int(row['z']):>6}  "
                f"{row['var']:>10.4f}  {row['tau_0_to_tc']:>12.4f}  "
                f"{row['tau_fit_raw']:>12.4f}  {row['tau_total']:>12.4f}  "
                f"{row['D']:>12.4f}\n"
            )
        f.write("=" * 90 + "\n\n")
        valid = df.dropna(subset=['D'])
        f.write("Statistics (valid D values only):\n")
        f.write(f"  Mean D   : {valid['D'].mean():.4f} ± {valid['D'].std():.4f}\n")
        f.write(f"  Min  D   : {valid['D'].min():.4f}\n")
        f.write(f"  Max  D   : {valid['D'].max():.4f}\n")
        f.write(f"  N valid  : {len(valid)} / {len(df)}\n")
    print(f"✓ Detailed results saved to: {txt_path}")
        # Two-column output: z and D only
    final_path = os.path.join(output_dir, "final_diffusion.txt")
    with open(final_path, 'w') as f:
        f.write(f"{'z':>6}  {'D':>14}\n")
        for _, row in df.iterrows():
            f.write(f"{int(row['z']):>6}  {row['D']:>14.6f}\n")
    print(f"✓ Final z vs D saved to:      {final_path}")
    return df


def make_plots(df, output_dir):
    """Generate and save four diagnostic plots plus a focused D-vs-z plot."""
    plot_df = df.dropna(subset=['D']).sort_values('z').copy()

    if len(plot_df) == 0:
        print("No valid D values — skipping plots.")
        return

    # ---- 4-panel summary ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Diffusion Coefficient Analysis', fontsize=16, fontweight='bold')

    ax = axes[0, 0]
    ax.plot(plot_df['z'], plot_df['D'], 'o-', lw=2, ms=7,
            color='#2E86AB', mfc='#A23B72', mew=2, mec='#2E86AB')
    ax.set_xlabel('z (window index)', fontsize=12, fontweight='bold')
    ax.set_ylabel('D  [Å²/lag]', fontsize=12, fontweight='bold')
    ax.set_title('Diffusion Coefficient vs Position', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, ls='--')

    ax = axes[0, 1]
    ax.plot(plot_df['z'], plot_df['tau_total'], 's-', lw=2, ms=7,
            color='#F18F01', mfc='#C73E1D', mew=2, mec='#F18F01')
    ax.set_xlabel('z (window index)', fontsize=12, fontweight='bold')
    ax.set_ylabel('τ_total  [Å²·lag]', fontsize=12, fontweight='bold')
    ax.set_title('Total Correlation Time vs Position', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, ls='--')

    ax = axes[1, 0]
    ax.bar(plot_df['z'], plot_df['tau_0_to_tc'], label='τ (0→tc)',
           color='#6A4C93', alpha=0.7, edgecolor='black', width=0.6)
    ax.bar(plot_df['z'], plot_df['tau_fit_raw'],
           bottom=plot_df['tau_0_to_tc'],
           label='τ_fit (tc→∞)', color='#1982C4', alpha=0.7,
           edgecolor='black', width=0.6)
    ax.set_xlabel('z (window index)', fontsize=12, fontweight='bold')
    ax.set_ylabel('τ  [Å²·lag]', fontsize=12, fontweight='bold')
    ax.set_title('Correlation Time Components', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, ls='--', axis='y')

    ax = axes[1, 1]
    ax.plot(plot_df['z'], plot_df['var'], '^-', lw=2, ms=7,
            color='#06A77D', mfc='#D7263D', mew=2, mec='#06A77D')
    ax.set_xlabel('z (window index)', fontsize=12, fontweight='bold')
    ax.set_ylabel('var(dz)  [Å²]', fontsize=12, fontweight='bold')
    ax.set_title('Variance of dz vs Position', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, ls='--')

    for a in axes.flat:
        a.tick_params(labelsize=10)
    plt.tight_layout()
    path = os.path.join(output_dir, 'diffusion_analysis.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ 4-panel plot saved: {path}")

    # ---- Focused D vs z ----
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(plot_df['z'], plot_df['D'], 'o-', lw=2.5, ms=9,
            color='#2E86AB', mfc='#A23B72', mew=2.5, mec='#2E86AB',
            label='D per window')
    mean_D = plot_df['D'].mean()
    ax.axhline(mean_D, color='red', ls='--', lw=2, alpha=0.7,
               label=f'Mean D = {mean_D:.4f}')
    ax.set_xlabel('z (window index)', fontsize=14, fontweight='bold')
    ax.set_ylabel('D = var² / τ_total  [Å²/lag]', fontsize=14, fontweight='bold')
    ax.set_title('Diffusion Coefficient vs Position', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3, ls='--')
    ax.legend(fontsize=12)
    ax.tick_params(labelsize=12)
    plt.tight_layout()
    path = os.path.join(output_dir, 'diffusion_vs_z.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ D-vs-z plot saved: {path}")


def print_statistics(df):
    valid = df.dropna(subset=['D'])
    print("\n" + "=" * 60)
    print("Final Statistics")
    print("=" * 60)
    print(f"  Windows processed  : {len(df)}")
    print(f"  Valid D values     : {len(valid)}")
    print(f"  Mean D             : {valid['D'].mean():.4f} ± {valid['D'].std():.4f}")
    print(f"  Min  D             : {valid['D'].min():.4f}  (z={valid.loc[valid['D'].idxmin(), 'z']})")
    print(f"  Max  D             : {valid['D'].max():.4f}  (z={valid.loc[valid['D'].idxmax(), 'z']})")
    print(f"  Mean tau_total     : {valid['tau_total'].mean():.4f} ± {valid['tau_total'].std():.4f}")
    print("=" * 60)



# MAIN

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("Step 1: Reading trajectory files")
    print("=" * 60)
    data_dict = read_data(DATA_DIR, FILE_RANGE, skiprows=SKIPROWS)

    print("Step 2: Computing dz = z - mean(z)")
    dz_dict = compute_dz(data_dict)

    print("\nStep 3: Computing FFT autocorrelation C(t)")
    C_fft_dict = compute_fft_acf(dz_dict)

    print("\nStep 4: Finding tc (first negative crossing) and integrating 0→tc")
    tc_tau_dict = find_tc_and_tau(C_fft_dict)

    print("\nStep 5: Fitting damped cosine to tail (tc→∞)")
    fit_params_dict = fit_tail(C_fft_dict, tc_tau_dict, dz_dict)

    print("\nStep 6: Computing diffusion coefficients")
    results = compute_diffusion(C_fft_dict, tc_tau_dict, fit_params_dict, dz_dict)

    print("\nStep 7: Saving results")
    df = save_results(results, OUTPUT_DIR)

    print("\nStep 8: Generating plots")
    make_plots(df, OUTPUT_DIR)

    print_statistics(df)
    print("\nDone.")
