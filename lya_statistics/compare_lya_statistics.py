"""
Compare the Lya statistics (mean transmitted flux and flux power spectrum) written
in the analysis files of two Cholla runs, e.g. one where the transmitted flux was
computed on the CPU (baseline) and one where it was computed on the GPU (GPU_LYA_FLUX).

The phase diagram stored in the same analysis files does not depend on how the
Lya flux is computed, so it is used to detect whether the gas fields of the two
runs are identical at each snapshot:
  - identical gas:  the Lya statistics must agree to round-off (strict tolerance).
  - diverged gas:   the runs evolved slightly differently (e.g. non-deterministic
                    GPU reductions), so only physical agreement is required
                    (relative difference of the effective optical depth and P(k)).

Usage:
  python compare_lya_statistics.py [--dir_a DIR] [--dir_b DIR] [--output_dir DIR]
"""

import os, sys, argparse
import numpy as np

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/'
sys.path.append(root_dir + 'tools')
from load_data import load_analysis_data

# load_analysis_data clamps the mean flux to this value
F_MIN = 1e-60

work_dir = os.environ.get('work', os.path.expanduser('~/work'))
default_base_dir = f'{work_dir}/benchmarks/cholla/cosmology/gpu_lya_spectra/'


def get_args():
  parser = argparse.ArgumentParser(description='Compare the Lya statistics of two Cholla runs.')
  parser.add_argument('--dir_a', default=default_base_dir + 'baseline/run_nnodes1/analysis/',
                      help='Analysis directory of the reference run (CPU Lya flux)')
  parser.add_argument('--dir_b', default=default_base_dir + 'gpu_spectra/run_nnodes1/analysis/',
                      help='Analysis directory of the run to check (GPU Lya flux)')
  parser.add_argument('--label_a', default='CPU flux')
  parser.add_argument('--label_b', default='GPU flux')
  parser.add_argument('--output_dir', default=None, help='Where to save the figure (default: dir_b/..)')
  parser.add_argument('--rtol_strict', type=float, default=1e-8,
                      help='Relative tolerance when the gas of both runs is identical')
  parser.add_argument('--rtol_tau', type=float, default=1e-2,
                      help='Relative tolerance on tau_eff=-ln(F_mean) when the gas has diverged')
  parser.add_argument('--rtol_pk', type=float, default=5e-2,
                      help='Relative tolerance on the flux P(k) when the gas has diverged')
  parser.add_argument('--no_plot', action='store_true')
  return parser.parse_args()


def get_snapshot_ids(input_dir):
  ids = [int(f.split('_')[0]) for f in os.listdir(input_dir) if f.endswith('_analysis.h5')]
  return sorted(ids)


def relative_diff(a, b):
  a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
  denom = np.where(a == 0, 1.0, np.abs(a))
  return np.where((a == 0) & (b == 0), 0.0, np.abs(b - a) / denom)


def compare_mean_flux(F_a, F_b, gas_identical, args):
  """Return (relative difference used for the check, tolerance, passed)."""
  if F_a <= F_MIN and F_b <= F_MIN: return 0.0, 0.0, True  # Both fully absorbed
  if gas_identical:
    rdiff = float(relative_diff(F_a, F_b))
    return rdiff, args.rtol_strict, rdiff <= args.rtol_strict
  tau_a, tau_b = -np.log(F_a), -np.log(F_b)
  rdiff = float(relative_diff(tau_a, tau_b))
  return rdiff, args.rtol_tau, rdiff <= args.rtol_tau


def compare_power_spectrum(ps_a, ps_b, gas_identical, args):
  """Return (k_vals, P_a, P_b, max relative difference, tolerance, passed)."""
  k_a, k_b = ps_a['k_vals'], ps_b['k_vals']
  tol = args.rtol_strict if gas_identical else args.rtol_pk
  if k_a is None or k_b is None or len(k_a) == 0 or len(k_b) == 0:
    # P(k) is only written when there is transmitted flux
    passed = (k_a is None or len(k_a) == 0) and (k_b is None or len(k_b) == 0)
    return None, None, None, np.nan, tol, passed
  k_vals, ids_a, ids_b = np.intersect1d(k_a, k_b, return_indices=True)
  P_a, P_b = ps_a['ps_mean'][ids_a], ps_b['ps_mean'][ids_b]
  max_rdiff = float(relative_diff(P_a, P_b).max())
  passed = (len(k_vals) == len(k_a) == len(k_b)) and max_rdiff <= tol
  return k_vals, P_a, P_b, max_rdiff, tol, passed


def compare_runs(args):
  ids_a, ids_b = get_snapshot_ids(args.dir_a), get_snapshot_ids(args.dir_b)
  snapshot_ids = sorted(set(ids_a) & set(ids_b))
  if len(snapshot_ids) == 0:
    print('ERROR: No common analysis files found.')
    sys.exit(1)
  missing = sorted(set(ids_a) ^ set(ids_b))
  if missing: print(f'WARNING: Snapshots present in only one of the runs: {missing}')

  print(f'Run A ({args.label_a}): {args.dir_a}')
  print(f'Run B ({args.label_b}): {args.dir_b}\n')
  header = f"{'id':>3} {'z':>7} {'gas':>8} | {'F_HI':>11} {'rdiff':>9} | {'F_HeII':>11} {'rdiff':>9} | {'P(k) max rdiff':>14} | status"
  print(header)
  print('-' * len(header))

  results = []
  all_passed = True
  for n_file in snapshot_ids:
    data_a = load_analysis_data(n_file, args.dir_a, phase_diagram=True)
    data_b = load_analysis_data(n_file, args.dir_b, phase_diagram=True)
    lya_a, lya_b = data_a['lya_statistics'], data_b['lya_statistics']
    z_a, z_b = data_a['cosmology']['current_z'], data_b['cosmology']['current_z']

    gas_identical = np.array_equal(data_a['phase_diagram']['data'], data_b['phase_diagram']['data'])
    same_setup = (lya_a['n_skewers'] == lya_b['n_skewers']) and np.isclose(z_a, z_b, rtol=1e-10, atol=0)

    F_HI   = compare_mean_flux(lya_a['Flux_mean'], lya_b['Flux_mean'], gas_identical, args)
    F_HeII = compare_mean_flux(lya_a['Flux_mean_HeII'], lya_b['Flux_mean_HeII'], gas_identical, args)
    k_vals, P_a, P_b, pk_rdiff, pk_tol, pk_passed = compare_power_spectrum(
        lya_a['power_spectrum'], lya_b['power_spectrum'], gas_identical, args)

    passed = same_setup and F_HI[2] and F_HeII[2] and pk_passed
    all_passed &= passed

    status = 'PASS' if passed else 'FAIL'
    if not same_setup:
      status += f" (n_skewers {lya_a['n_skewers']} vs {lya_b['n_skewers']}, z {z_a} vs {z_b})"
    gas_str = 'same' if gas_identical else 'diverged'
    print(f"{n_file:3d} {z_a:7.3f} {gas_str:>8} | {lya_a['Flux_mean']:11.4e} {F_HI[0]:9.2e} | "
          f"{lya_a['Flux_mean_HeII']:11.4e} {F_HeII[0]:9.2e} | {pk_rdiff:14.2e} | {status}")

    results.append({'n_file': n_file, 'z': z_a, 'gas_identical': gas_identical,
                    'F_HI_a': lya_a['Flux_mean'], 'F_HI_b': lya_b['Flux_mean'],
                    'F_HeII_a': lya_a['Flux_mean_HeII'], 'F_HeII_b': lya_b['Flux_mean_HeII'],
                    'F_HI_rdiff': F_HI[0], 'F_HeII_rdiff': F_HeII[0],
                    'k_vals': k_vals, 'P_a': P_a, 'P_b': P_b, 'passed': passed})

  print('\nTolerances:')
  print(f'  gas same:     relative difference of F_mean and P(k) <= {args.rtol_strict:.1e}')
  print(f'  gas diverged: relative difference of tau_eff=-ln(F_mean) <= {args.rtol_tau:.1e}, '
        f'of P(k) <= {args.rtol_pk:.1e}')
  print('  (gas "diverged" means the phase diagrams differ, i.e. the runs evolved differently,')
  print('   so differences there are not caused by the Lya flux computation alone)')
  print(f"\nResult: {'ALL SNAPSHOTS PASSED' if all_passed else 'SOME SNAPSHOTS FAILED'}")
  return results, all_passed


def plot_comparison(results, args):
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt

  output_dir = args.output_dir
  if output_dir is None: output_dir = os.path.dirname(os.path.normpath(args.dir_b)) + '/'
  os.makedirs(output_dir, exist_ok=True)

  z = np.array([r['z'] for r in results])
  fig, axes = plt.subplots(1, 3, figsize=(18, 5))

  ax = axes[0]
  for key, name, color in [('HI', 'HI', 'C0'), ('HeII', 'HeII', 'C1')]:
    F_a = np.array([r[f'F_{key}_a'] for r in results])
    F_b = np.array([r[f'F_{key}_b'] for r in results])
    valid = (F_a > F_MIN) & (F_b > F_MIN)
    ax.plot(z[valid], F_a[valid], 'o-', color=color, label=f'{name} {args.label_a}')
    ax.plot(z[valid], F_b[valid], 'x--', color=color, ms=8, label=f'{name} {args.label_b}')
  ax.set_yscale('log')
  ax.set_xlabel('z')
  ax.set_ylabel(r'$\langle F \rangle$')
  ax.set_title('Mean transmitted flux')
  ax.legend()

  ax = axes[1]
  gas_same = np.array([r['gas_identical'] for r in results])
  for key, name, color in [('HI', 'HI', 'C0'), ('HeII', 'HeII', 'C1')]:
    rdiff = np.array([r[f'F_{key}_rdiff'] for r in results])
    rdiff = np.maximum(rdiff, 1e-17)
    ax.plot(z, rdiff, '-', color=color, alpha=0.5)
    ax.plot(z[gas_same], rdiff[gas_same], 's', color=color, label=rf'{name}: $|\Delta F|/F$ (gas same)')
    ax.plot(z[~gas_same], rdiff[~gas_same], 'o', color=color, mfc='none',
            label=rf'{name}: $|\Delta\tau|/\tau$ (gas diverged)')
  ax.axhline(args.rtol_strict, color='k', ls=':', label='strict tolerance')
  ax.axhline(args.rtol_tau, color='k', ls='--', label=r'$\tau$ tolerance')
  ax.set_yscale('log')
  ax.set_xlabel('z')
  ax.set_ylabel('relative difference')
  ax.set_title(f'{args.label_b} vs {args.label_a}')
  ax.legend(fontsize=8)

  ax = axes[2]
  with_pk = [r for r in results if r['k_vals'] is not None]
  cmap = plt.get_cmap('viridis')
  for i, r in enumerate(with_pk):
    color = cmap(i / max(len(with_pk) - 1, 1))
    ax.plot(r['k_vals'], r['P_b'] / r['P_a'] - 1, color=color, label=f"z={r['z']:.1f}")
  ax.axhline(0, color='k', lw=0.5)
  ax.axhspan(-args.rtol_pk, args.rtol_pk, color='gray', alpha=0.15)
  ax.set_xscale('log')
  ax.set_xlabel(r'$k$  [s/km]')
  ax.set_ylabel(rf'$P_{{\rm {args.label_b}}}(k) / P_{{\rm {args.label_a}}}(k) - 1$')
  ax.set_title('Flux power spectrum')
  ax.legend(fontsize=7, ncol=2)

  fig.tight_layout()
  figure_name = output_dir + 'compare_lya_statistics.png'
  fig.savefig(figure_name, dpi=150)
  print(f'Saved figure: {figure_name}')


if __name__ == '__main__':
  args = get_args()
  results, all_passed = compare_runs(args)
  if not args.no_plot: plot_comparison(results, args)
  sys.exit(0 if all_passed else 1)
