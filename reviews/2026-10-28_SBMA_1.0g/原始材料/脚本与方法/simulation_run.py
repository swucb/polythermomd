"""SBMA/CBMA: bounded equilibration followed by 10 ns; independent of node-2 files."""
from pathlib import Path
import datetime, fcntl, json, math, os, shutil, subprocess, time

B = Path(__file__).resolve().parent
SOURCE = B.parent / 'network_opls_resp_spce_20261002'
G = '${WORKSTATION_PROJECT}/software/gromacs-2025.4/bin/gmx'
IDS = ['SBMA_1.0g', 'CBMA_1.0g', 'SBMA_0.5g', 'CBMA_0.5g', 'SBMA_1.5g', 'CBMA_1.5g']
ENV = dict(os.environ, OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
BASE = '''integrator = md
dt = 0.0005
cutoff-scheme = Verlet
coulombtype = PME
rcoulomb = 1.2
rvdw = 1.2
rlist = 1.4
verlet-buffer-tolerance = -1
nstlist = 10
DispCorr = EnerPres
pbc = xyz
periodic-molecules = yes
constraints = h-bonds
lincs-order = 6
lincs-iter = 2
tcoupl = V-rescale
tc-grps = System
tau-t = 1.0
ref-t = 298
nstlog = 20000
nstenergy = 2000
nstcalcenergy = 100
nstxout = 0
nstvout = 0
nstfout = 0
compressed-x-precision = 1000
'''

def dump(p, obj):
    temp = p.with_suffix('.tmp')
    temp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n')
    temp.replace(p)

def gpu_busy():
    out = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name', '--format=csv,noheader'], text=True)
    return any(line.strip() and 'gnome-remote-desktop' not in line for line in out.splitlines())

def resources(d, state):
    while True:
        available = int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:'))) / 1024**2
        free = shutil.disk_usage(B).free / 1024**3
        if free < 100:
            raise RuntimeError('Less than 100 GiB disk free; preserve outputs and inspect.')
        if not gpu_busy() and available >= 6:
            return
        state.update(status='waiting_for_resources', available_memory_GiB=available)
        dump(d / 'state.json', state)
        time.sleep(30)

def command(d, args, logfile, stdin=None):
    with (d / logfile).open('w') as f:
        subprocess.run([G] + args, cwd=d, env=ENV, input=stdin, text=True,
                       stdout=f, stderr=subprocess.STDOUT, check=True)

def stage(d, name, mdp, coord, checkpoint, state):
    # Completed stages are not rerun; partial stages resume from their checkpoint.
    if (d / (name + '.log')).exists() and 'Finished mdrun' in (d / (name + '.log')).read_text():
        return
    resources(d, state)
    state.update(status=name + '_running', stage_started=datetime.datetime.now().isoformat())
    dump(d / 'state.json', state)
    p = d / (name + '.mdp')
    if p.exists() and p.read_text() != mdp:
        raise RuntimeError('Existing MDP differs: ' + str(p))
    p.write_text(mdp)
    if not (d / (name + '.tpr')).exists():
        args = ['grompp', '-f', p.name, '-c', coord, '-p', 'topol.top', '-o', name + '.tpr', '-maxwarn', '0']
        if checkpoint:
            args += ['-t', checkpoint]
        command(d, args, name + '_grompp.log')
    args = ['mdrun', '-deffnm', name, '-ntmpi', '1', '-ntomp', '4', '-nb', 'gpu', '-pme', 'gpu', '-bonded', 'cpu', '-update', 'cpu', '-cpt', '10']
    if (d / (name + '.cpt')).exists():
        args += ['-cpi', name + '.cpt', '-append']
    command(d, args, name + '_console.log')
    log = (d / (name + '.log')).read_text()
    if 'Finished mdrun' not in log or 'LINCS WARNING' in log or 'Fatal error' in log:
        raise RuntimeError('Stage failed or constraint warning: ' + name)

def energy(d, name):
    command(d, ['energy', '-f', name + '.edr', '-o', name + '_metrics.xvg'], name + '_energy.log', 'Temperature\nDensity\nPotential\n0\n')
    rows = [[float(v) for v in line.split()] for line in (d / (name + '_metrics.xvg')).read_text().splitlines() if line.strip() and not line.startswith(('#', '@'))]
    if not rows or any(not math.isfinite(v) for row in rows for v in row):
        raise RuntimeError('Missing/nonfinite equilibration energies')
    # Only the latter half of each 1 ns block is used for the operational gate.
    rows = rows[len(rows)//2:]
    result = {}
    header = (d / (name + '_metrics.xvg')).read_text()
    import re
    labels = re.findall(r'@\s+s\d+\s+legend\s+"([^"]+)"', header)
    for i, label in enumerate(labels, 1):
        result[label] = sum(row[i] for row in rows) / len(rows)
    if not all(k in result for k in ('Temperature', 'Density', 'Potential')):
        raise RuntimeError('Unexpected energy selections: ' + repr(labels))
    return result

def one(ident):
    d = B / ident
    d.mkdir(exist_ok=True)
    p = d / 'state.json'
    state = json.loads(p.read_text()) if p.exists() else dict(id=ident, production_ns=10, temperature_K=298, dt_ps=0.0005, independent_replica=1,
        source=str(SOURCE / ident), physical_validation=False, authorized='User 2026-10-27: start SBMA and CBMA, 10 ns')
    state['production_ns'] = 10
    state['duration_update'] = 'User 2026-10-27: reduce production to 10 ns; retain equilibration progress'
    if state.get('status') == 'production_complete':
        return state
    try:
        for name in ['topol.top', 'network.itp', 'nve.gro']:
            dest = d / ('start.gro' if name == 'nve.gro' else name)
            if not dest.exists():
                shutil.copy2(SOURCE / ident / name, dest)
        stage(d, 'eq_nvt', BASE + 'nsteps = 200000\npcoupl = no\nnstxout-compressed = 2000\ncontinuation = no\ngen-vel = yes\ngen-temp = 298\ngen-seed = ' + str(20261027 + IDS.index(ident)) + '\n', 'start.gro', None, state)
        metrics = []
        previous = 'eq_nvt'
        accepted = False
        for block in range(1, 11):
            name = f'eq_npt_{block:02d}'
            mdp = BASE + 'nsteps = 2000000\nnstxout-compressed = 2000\ncontinuation = yes\ngen-vel = no\npcoupl = C-rescale\npcoupltype = isotropic\ntau-p = 5.0\nref-p = 1.0\ncompressibility = 4.5e-5\n'
            stage(d, name, mdp, previous + '.gro', previous + '.cpt', state)
            metrics.append(energy(d, name))
            previous = name
            state['equilibration_block_means'] = metrics
            dump(p, state)
            if block >= 5:
                last = metrics[-3:]
                temps_ok = all(abs(x['Temperature'] - 298) < 3 for x in last)
                density_span = (max(x['Density'] for x in last) - min(x['Density'] for x in last)) / last[-1]['Density']
                potential_span = (max(x['Potential'] for x in last) - min(x['Potential'] for x in last)) / max(abs(last[-1]['Potential']), 1)
                state['operational_gate'] = dict(temperature_ok=temps_ok, density_relative_span=density_span, potential_relative_span=potential_span,
                    note='Operational stationarity screen, not proof of full structural equilibration or physical validation')
                dump(p, state)
                if temps_ok and density_span < 0.01 and potential_span < 0.01:
                    accepted = True
                    break
        if not accepted:
            raise RuntimeError('NPT stationarity screen not met after 10 ns; hold production for inspection')
        state['production_volume_source'] = previous + '.gro'
        dump(p, state)
        stage(d, 'prod_10ns', BASE + 'nsteps = 20000000\nnstxout-compressed = 200\ncontinuation = yes\ngen-vel = no\npcoupl = no\n', previous + '.gro', previous + '.cpt', state)
        state.update(status='production_complete', completed=datetime.datetime.now().isoformat())
    except Exception as exc:
        state.update(status='needs_attention', error=str(exc))
    dump(p, state)
    return state

if __name__ == '__main__':
    lock = open('${WORKSTATION_PROJECT}/md_pilot_gpu.lock', 'a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (B / 'library').mkdir(exist_ok=True)
    for src in (SOURCE / 'library').glob('*.itp'):
        dest = B / 'library' / src.name
        if not dest.exists():
            shutil.copy2(src, dest)
    results = []
    for ident in IDS:
        result = one(ident)
        results.append(result)
        dump(B / 'summary.json', dict(results=results, queue=IDS, production_ns=10, automatic_extension=False))
        print(ident, result['status'], result.get('error', ''), flush=True)
        if result['status'] == 'needs_attention':
            break
