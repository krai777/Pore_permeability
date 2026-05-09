import numpy as np
import pandas as pd
from pathlib import Path
import os
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

directory_path = "path of your z coordinates"  # Change this to your actual path

data_dict = {}

for i in range(-30, 31):
    filename = os.path.join(directory_path, f"z1_{i}.dat")
    data = np.loadtxt(filename, skiprows=1)
    data_dict[i] = data
    print(f"Successfully read {filename}: {len(data)} rows")

for i in data_dict.keys():
    z = data_dict[i][:, 1]
    z_mean = np.mean(z)
    dz = z - z_mean
    dz_dict[i] = dz
    print(f"File z1_{i}.dat: mean(z) = {z_mean:.4f} Å, dz range = [{np.min(dz):.4f}, {np.max(dz):.4f}] Å")
print(f"\nCalculated dz for {len(dz_dict)} files")

C_fft_dict = {}

for i in dz_dict.keys():
    dz = dz_dict[i]
    N = len(dz)

    dz_padded = np.concatenate([dz, np.zeros(N-1)])

    F_dz = np.fft.fft(dz_padded)

    power_spectrum = np.abs(F_dz)**2

    IFT = np.fft.ifft(power_spectrum).real

    C_fft = np.zeros(N)
    for m in range(N):
        C_fft[m] = IFT[m] / (N - m)

    C_fft_dict[i] = C_fft

    print(f"File z1_{i}.dat: C_fft(0) = {C_fft[0]:.4f}")

print(f"\nComputed C(m) via FFT method for {len(C_fft_dict)} files")

# Create directory for FFT method plots
plot_fft_z_dir = "plot_fft_z"
os.makedirs(plot_fft_z_dir, exist_ok=True)
# Dictionary to store tc and tau values
tc_tau_dict = {}
D_tc_dict = {}
tc_inf_dict = {}

print("Computing tc (first negative crossing) and tau (integral) for each file...\n")

for i in sorted(C_fft_dict.keys()):
    C_fft_m = C_fft_dict[i]

    negative_indices = np.where(C_fft_m < 0)[0]

    if len(negative_indices) > 0:
        tc = negative_indices[0]
    else:
        print('no')
        tc = len(C_fft_m) - 1
        print(f"Warning: File z1_{i}.dat - C(m) never becomes negative. Using tc = {tc}")
    var = np.var(dz_dict[i])
    
    m_values = np.arange(tc + 1)
    C_values = C_fft_m[:tc + 1]
    tau = np.trapz(C_values, m_values)
    tc_tau_dict[i] = {'tc': tc, 'tau': tau}
    print(f"File z1_{i}.dat: tc = {tc}, tau = {tau:.4f} Ų·units")

# Create DataFrame for better visualization and saving
results_df = pd.DataFrame.from_dict(tc_tau_dict, orient='index')
results_df.index.name = 'file_index'
results_df = results_df.reset_index()

print("\n" + "="*60)
print("Summary of Results:")
print("="*60)
print(results_df.to_string(index=False))

output_csv = "tc_tau_results.csv"
results_df.to_csv(output_csv, index=False)
print(f"\n✓ Results saved to: {output_csv}")

output_txt = "tc_tau_results.txt"
with open(output_txt, 'w') as f:
    f.write("Correlation Length Analysis Results\n")
    f.write("="*60 + "\n\n")
    f.write("tc: First lag where C(m) becomes negative\n")
    f.write("tau: Integral of C(m) from m=0 to m=tc\n\n")
    f.write("="*60 + "\n")
    f.write(f"{'File Index':<12} {'tc':<10} {'tau (Ų·units)':<15}\n")
    f.write("="*60 + "\n")

    for idx, row in results_df.iterrows():
        f.write(f"{int(row['file_index']):<12} {int(row['tc']):<10} {row['tau']:<15.4f}\n")

    f.write("="*60 + "\n")
    f.write(f"\nTotal files processed: {len(tc_tau_dict)}\n")
    f.write(f"Mean tc: {results_df['tc'].mean():.2f}\n")
    f.write(f"Mean tau: {results_df['tau'].mean():.4f} Ų·units\n")
    f.write(f"Std tau: {results_df['tau'].std():.4f} Ų·units\n")

print(f"✓ Detailed results saved to: {output_txt}")

def damped_cosine(t, A, gamma, omega):
    """
    Damped cosine function: C(t) = A * exp(-gamma * t) * cos(omega * t)
    """
    return A * np.exp(-gamma * t) * np.cos(omega * t)
