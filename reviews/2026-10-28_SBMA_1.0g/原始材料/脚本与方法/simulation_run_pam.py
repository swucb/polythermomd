"""PAM baseline: wait for the existing GPU queue, then identical equilibration + 10 ns."""
import datetime
import fcntl
from pathlib import Path
import run as workflow

if __name__ == '__main__':
    workflow.IDS.append('pam_control')
    d = workflow.B / 'pam_control'
    d.mkdir(exist_ok=True)
    p = d / 'state.json'
    if not p.exists():
        workflow.dump(p, dict(id='pam_control', status='queued_waiting_for_existing_batch',
            production_ns=10, temperature_K=298, independent_replica=1,
            physical_validation=False, authorized='User 2026-10-28: PAM base also run',
            queued=datetime.datetime.now().isoformat()))
    lock = open('${WORKSTATION_PROJECT}/md_pilot_gpu.lock', 'a')
    fcntl.flock(lock, fcntl.LOCK_EX)
    result = workflow.one('pam_control')
    workflow.dump(workflow.B / 'pam_summary.json', result)
    print('pam_control', result['status'], result.get('error', ''), flush=True)
