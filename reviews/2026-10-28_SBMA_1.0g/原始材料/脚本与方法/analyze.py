"""Reproducible SBMA single-trajectory analysis; positions in Angstrom internally."""
from pathlib import Path
import os
os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
import json, subprocess, hashlib, time
import numpy as np
import MDAnalysis as mda
from MDAnalysis.lib.distances import capped_distance
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

B=Path(__file__).resolve().parent
D=Path('${WORKSTATION_PROJECT}/production_sbma_cbma_20261027/SBMA_1.0g')
O=B/'results';O.mkdir(exist_ok=True)
G='${WORKSTATION_PROJECT}/software/gromacs-2025.4/bin/gmx'
S=json.loads((B/'selections.json').read_text())

def savecsv(name, cols, data):
    np.savetxt(O/(name+'.csv'), data, delimiter=',', header=','.join(cols), comments='')

def mark(message):
    print(time.strftime('%H:%M:%S'),message,flush=True)

def energy():
    allrows=[]
    for stage in ['eq_nvt']+[f'eq_npt_{i:02d}' for i in range(1,6)]+['prod_10ns']:
        target=O/(stage+'_energy.xvg')
        with (O/(stage+'_energy_extract.log')).open('w') as f:
            subprocess.run([G,'energy','-f',str(D/(stage+'.edr')),'-o',str(target)],input='Temperature\nDensity\nPotential\nPressure\n0\n',text=True,stdout=f,stderr=subprocess.STDOUT,check=True)
        import re
        labels=re.findall(r'@\s+s\d+\s+legend\s+"([^"]+)"',target.read_text())
        a=np.loadtxt(target,comments=('#','@'))
        if 'Density' not in labels:
            snapshot=mda.Universe(str(D/'prod_10ns.tpr'),str(D/(stage+'.gro')))
            density=snapshot.atoms.masses.sum()*1660.53906660/snapshot.dimensions[:3].prod()
            a=np.column_stack([a,np.full(len(a),density)]);labels.append('Density')
        a=a[:,[0]+[1+labels.index(k) for k in ['Temperature','Density','Potential','Pressure']]]
        savecsv(stage+'_energy',['time_ps','temperature_K','density_kg_m3','potential_kJ_mol','pressure_bar'],a)
        allrows.append((stage,a))
    return allrows

def msd_fft(x):
    # Multi-origin MSD averaged across particles. No origin subsampling.
    x=np.asarray(x,dtype=np.float64);x=x-x[0]
    n=len(x);den=np.arange(n,0,-1,dtype=float)
    sq=np.mean(np.sum(x*x,axis=2),axis=1)
    corr=np.zeros(n)
    for start in range(0,x.shape[1],128):
        chunk=x[:,start:start+128]
        f=np.fft.rfft(chunk,n=2*n,axis=0)
        corr+=np.fft.irfft(f*f.conj(),n=2*n,axis=0)[:n].sum(axis=(1,2))/x.shape[1]
    cs=np.r_[0,np.cumsum(sq)]
    out=(cs[n-np.arange(n)]+cs[n]-cs[np.arange(n)]-2*corr)/den
    out[0]=0
    return np.maximum(out,0)

def fit_msd(y,dt,lo,hi):
    t=np.arange(len(y))*dt
    good=(t>=lo)&(t<=hi)&(y>0)
    slope,intercept=np.polyfit(t[good],y[good],1)
    residual=np.sum((y[good]-(slope*t[good]+intercept))**2)
    total=np.sum((y[good]-y[good].mean())**2)
    alpha=np.polyfit(np.log(t[good]),np.log(y[good]),1)[0]
    return dict(D_1e9_m2_s=float(slope/6),intercept_nm2=float(intercept),R2=float(1-residual/total),alpha=float(alpha),fit_ns=[lo,hi])

class Events:
    def __init__(self):
        self.active={};self.closed=[];self.censored=[];self.count=[];self.leftclosed=[];self.initialcount=0
    def step(self,keys,frame):
        keys=set(int(x) for x in keys)
        if frame==0:self.initialcount=len(keys)
        for key in self.active.keys()-keys:
            start=self.active.pop(key)
            # Events already present at the first frame are left-censored and excluded.
            if start>0:self.closed.append(frame-start)
            else:self.leftclosed.append(frame)
        for key in keys-self.active.keys():self.active[key]=frame
        self.count.append(len(keys))
    def finish(self,n):
        self.censored=[n-start for start in self.active.values() if start>0]
        return np.asarray(self.closed,dtype=int),np.asarray(self.censored,dtype=int)

def km(events,censored,dt):
    event=np.bincount(events) if len(events) else np.zeros(1)
    censor=np.bincount(censored) if len(censored) else np.zeros(1)
    n=max(len(event),len(censor));event=np.pad(event,(0,n-len(event)));censor=np.pad(censor,(0,n-len(censor)))
    risk=len(events)+len(censored);survival=1.;rows=[[0.,1.,risk]]
    for i in range(1,n):
        if risk:
            survival*=1-event[i]/risk
            rows.append([i*dt,survival,risk]);risk-=event[i]+censor[i]
    return np.asarray(rows)

def pairs(a,b,box,cut):
    p,dist=capped_distance(a,b,max_cutoff=cut,box=box,return_distances=True)
    return p,dist

def main():
    energies=energy();mark('energies extracted')
    u=mda.Universe(str(D/'prod_10ns.tpr'),str(D/'prod_10ns.xtc'))
    assert len(u.atoms)==int((D/'prod_10ns.gro').read_text().splitlines()[1])
    groups={k:u.select_atoms(v) for k,v in {'F3':'resname FC3 and name FE','F4':'resname FC4 and name FE','N3':'resname FC3 and name N*','N4':'resname FC4 and name N*','K':'resname K','W':'resname SOL and name OW'}.items()}
    groups['Pplus']=u.atoms[S['Nplus']];groups['SO']=u.atoms[S['sulfonateO']]
    assert len(groups['F3'])==62 and len(groups['F4'])==62 and len(groups['K'])==434
    assert len(groups['N3'])==372 and len(groups['N4'])==372
    assert len(u.atoms[:S['polymer_atoms']])==8714
    report=dict(system='SBMA_1.0g',n_atoms=len(u.atoms),frames=len(u.trajectory),raw_dt_ps=float(u.trajectory.dt),
        counts={k:len(v) for k,v in groups.items()},single_network_single_trajectory=True,
        rdf_sampling_ps=10,transport_contact_sampling_ps=1,
        limitations=['One network and one trajectory; block SD is not an independent-replicate confidence interval.',
        'Continuous contact lifetimes resolved at 1 ps; shorter interruptions can be missed.',
        'RDF uses global number-density normalization in an inhomogeneous gel.',
        'Bulk transport estimates do not predict electrode kinetics, I-V or device response time.'])
    rdfdefs={'N3_W':('N3','W'),'N4_W':('N4','W'),'N3_K':('N3','K'),'N4_K':('N4','K'),
        'N3_Pplus':('N3','Pplus'),'N4_Pplus':('N4','Pplus'),'F3_W':('F3','W'),'F4_W':('F4','W'),
        'Pplus_W':('Pplus','W'),'SO_W':('SO','W')}
    edges=np.arange(0,12.00001,.05);r=(edges[:-1]+edges[1:])/2
    hist={k:np.zeros(len(r)) for k in rdfdefs};normal={k:0. for k in rdfdefs};vols=[]
    cache=O/'rdf_cache.npz'
    if cache.exists():
        cached=np.load(cache);vols=cached['vols'].tolist()
        hist={k:cached['hist_'+k] for k in rdfdefs}
        normal={k:float(cached['norm_'+k]) for k in rdfdefs}
    else:
        for j,ts in enumerate(u.trajectory[::100]):
            assert np.allclose(ts.dimensions[3:],90)
            vols.append(ts.volume)
            for key,(a,b) in rdfdefs.items():
                pp,dd=pairs(groups[a].positions,groups[b].positions,ts.dimensions,12)
                hist[key]+=np.histogram(dd,edges)[0]
                normal[key]+=len(groups[a])*len(groups[b])/ts.volume
            if j%200==0:mark(f'RDF frame {j}/1001')
        np.savez(cache,vols=vols,**{'hist_'+k:v for k,v in hist.items()},**{'norm_'+k:v for k,v in normal.items()})
    shell=4*np.pi/3*np.diff(edges**3);rdf={k:hist[k]/(normal[k]*shell) for k in hist}
    report['volume_nm3']=float(np.mean(vols)/1000)
    # Search windows constrain the first-shell assignment to the chemical pair in question.
    windows={'N3_W':(2.4,3.5,4.5),'N4_W':(2.4,3.5,4.5),'N3_K':(2.4,3.7,5.2),'N4_K':(2.4,3.7,5.2),
        'N3_Pplus':(3.3,5.6,7.8),'N4_Pplus':(3.3,5.6,7.8),'Pplus_W':(3.5,5.2,6.8),'SO_W':(2.3,3.5,4.5)}
    for key,(a,b) in rdfdefs.items():
        cn=np.cumsum(hist[key])/(len(vols)*len(groups[a]))
        savecsv('rdf_'+key,['r_nm','g_r','coordination_per_reference_atom'],np.c_[r/10,rdf[key],cn])
    cuts={};cutinfo={}
    for key,(lo,peakmax,searchmax) in windows.items():
        smooth=gaussian_filter1d(rdf[key],2)
        peaks,_=find_peaks(smooth,prominence=.05)
        peaks=[p for p in peaks if lo<=r[p]<=peakmax]
        if not peaks:
            cutinfo[key]=dict(resolved_first_shell=False,note='No resolved first peak; use the paired oxidation-state geometric cutoff only for like-for-like proximity statistics')
            continue
        peak=peaks[0]
        minima,_=find_peaks(-smooth,prominence=.015)
        minima=[p for p in minima if r[p]>r[peak]+.2 and r[p]<=searchmax]
        if not minima:raise RuntimeError('No resolved RDF first minimum; inspect before defining contact: '+key)
        ix=minima[0];cuts[key]=float(r[ix]);cutinfo[key]=dict(peak_nm=float(r[peak]/10),cutoff_nm=float(r[ix]/10),smoothing_sigma_nm=.01)
    # Use common pair-type cutoffs for comparisons, the larger of the two resolved minima.
    for suffix in ['W','K','Pplus']:
        available=[cuts[k] for k in ['N3_'+suffix,'N4_'+suffix] if k in cuts]
        if not available:raise RuntimeError('Neither oxidation state has a resolved shell: '+suffix)
        common=max(available)
        cuts['N3_'+suffix]=cuts['N4_'+suffix]=common
    report['rdf_cutoffs']=cutinfo;report['common_cutoffs_nm']={k:v/10 for k,v in cuts.items()}
    for key,(a,b) in rdfdefs.items():
        cn=np.cumsum(hist[key])/(len(vols)*len(groups[a]))
        savecsv('rdf_'+key,['r_nm','g_r','coordination_per_reference_atom'],np.c_[r/10,rdf[key],cn])
    mark('RDF done; contact / transport stream starts')
    nf=(len(u.trajectory)-1)//10+1
    select=u.atoms[np.r_[groups['F3'].indices,groups['F4'].indices,groups['K'].indices,groups['W'].indices]]
    pos=np.empty((nf,len(select),3),np.float32);drifts=np.empty((nf,3));last=None;unwrapped=None;drift=np.zeros(3)
    masses=u.atoms.masses[:,None];totalmass=masses.sum()
    eventnames=['F3_K','F4_K','F3_Pplus','F4_Pplus','water_F3','water_F4','water_Pplus','water_SO']
    ev={k:Events() for k in eventnames};fractions=[];nwater=len(groups['W'])
    nK=len(groups['K']);nP=len(groups['Pplus']);nSO=len(groups['SO'])//3
    for frame,ts in enumerate(u.trajectory[::10]):
        xyz=u.atoms.positions.astype(float);box=ts.dimensions[:3]
        if last is None:unwrapped=xyz.copy()
        else:
            delta=xyz-last;delta-=box*np.rint(delta/box)
            unwrapped+=delta;drift+=(delta*masses).sum(axis=0)/totalmass
        last=xyz;drifts[frame]=drift
        pos[frame]=(unwrapped[select.indices]-drift)/10
        values=[]
        for label,nname in [('F3','N3'),('F4','N4')]:
            p,_=pairs(groups[nname].positions,groups['K'].positions,ts.dimensions,cuts[nname+'_K'])
            ids=np.unique((p[:,0]//6)*nK+p[:,1]) if len(p) else np.array([],int)
            ev[label+'_K'].step(ids,frame)
            values += [len(np.unique(ids//nK))/62,len(ids)/62]
            p,_=pairs(groups[nname].positions,groups['Pplus'].positions,ts.dimensions,cuts[nname+'_Pplus'])
            ids=np.unique((p[:,0]//6)*nP+p[:,1]) if len(p) else np.array([],int)
            ev[label+'_Pplus'].step(ids,frame)
            values += [len(np.unique(ids//nP))/62,len(ids)/62]
            p,_=pairs(groups[nname].positions,groups['W'].positions,ts.dimensions,cuts[nname+'_W'])
            ids=np.unique((p[:,0]//6)*nwater+p[:,1]) if len(p) else np.array([],int)
            ev['water_'+label].step(ids,frame)
            values += [len(ids)/62]
        p,_=pairs(groups['Pplus'].positions,groups['W'].positions,ts.dimensions,cuts['Pplus_W'])
        ev['water_Pplus'].step(np.unique(p[:,0]*nwater+p[:,1]),frame)
        p,_=pairs(groups['SO'].positions,groups['W'].positions,ts.dimensions,cuts['SO_W'])
        ev['water_SO'].step(np.unique((p[:,0]//3)*nwater+p[:,1]),frame)
        fractions.append([ts.time]+values)
        if frame%1000==0:mark(f'contact/transport frame {frame}/{nf}')
    cols=['time_ps']
    for lab in ['F3','F4']:cols += [lab+'_K_fraction',lab+'_K_number',lab+'_P_fraction',lab+'_P_number',lab+'_water_number']
    fractions=np.asarray(fractions);savecsv('contacts_timeseries',cols,fractions)
    report['contact_means']={col:float(fractions[:,i].mean()) for i,col in enumerate(cols) if i}
    report['contact_block_sd']={col:float(np.std([x[:,i].mean() for x in np.array_split(fractions,5)],ddof=1)) for i,col in enumerate(cols) if i}
    report['lifetimes']={}
    for name,tracker in ev.items():
        closed,censored=tracker.finish(nf);curve=km(closed,censored,1.)
        savecsv('survival_'+name,['time_ps','continuous_survival_KM','number_at_risk'],curve)
        savecsv('events_'+name,['duration_ps','right_censored'],np.r_[np.c_[closed,np.zeros(len(closed))],np.c_[censored,np.ones(len(censored))]])
        median=float(curve[np.flatnonzero(curve[:,1]<=.5)[0],0]) if np.any(curve[:,1]<=.5) else None
        left_remaining=sum(start==0 for start in tracker.active.values())
        residual=np.ones(nf)
        if tracker.initialcount:
            residual-=np.cumsum(np.bincount(tracker.leftclosed,minlength=nf))/tracker.initialcount
        savecsv('initial_cohort_'+name,['time_ps','continuous_remaining_survival'],np.c_[np.arange(nf),residual])
        report['lifetimes'][name]=dict(initial_contacts=tracker.initialcount,initial_contacts_remaining_at_end=left_remaining,
            initial_contacts_remaining_fraction=left_remaining/tracker.initialcount if tracker.initialcount else None,complete_events=len(closed),right_censored=len(censored),median_ps=median,
            final_survival=float(curve[-1,1]),mean_event_duration_ps=float(closed.mean()) if len(closed) else None,
            note='1 ps sampled continuous contact; same molecule/site pair; no gap tolerance; left-censored events excluded')
    mark('contacts complete; FFT MSD')
    ranges={'FC3':(0,62),'FC4':(62,124),'K':(124,558),'water':(558,len(select))}
    report['diffusion']={};curves={}
    for name,(a,b) in ranges.items():
        x=pos[:,a:b];y=msd_fft(x);curves[name]=y
        fit=fit_msd(y,.001,.5,2.)
        blockfits=[]
        for indices in np.array_split(np.arange(nf),5):
            blockfits.append(fit_msd(msd_fft(x[indices]),.001,.2,.8)['D_1e9_m2_s'])
        fit['block_D_1e9_m2_s']=blockfits;fit['block_SD']=float(np.std(blockfits,ddof=1))
        fit['fit_window_sensitivity']=[fit_msd(y,.001,lo,hi) for lo,hi in [(.2,1),(1,3),(2,4)]]
        fit['diffusive_regime_screen']=bool(.8<=fit['alpha']<=1.2 and fit['R2']>.98 and fit['D_1e9_m2_s']>0)
        report['diffusion'][name]=fit
        savecsv('msd_'+name,['lag_ns','MSD_nm2'],np.c_[np.arange(nf//2+1)*.001,y[:nf//2+1]])
        mark('MSD '+name+' done')
    q=-3*pos[:,:62].sum(axis=1)-4*pos[:,62:124].sum(axis=1)+pos[:,124:558].sum(axis=1)
    qmsd=msd_fft(q[:,None,:]);savecsv('collective_charge_msd',['lag_ns','charge_displacement_MSD_e2_nm2'],np.c_[np.arange(nf//2+1)*.001,qmsd[:nf//2+1]])
    tmean=float(energies[-1][1][:,1].mean());V=report['volume_nm3']*1e-27
    pref=1.602176634e-19**2/(V*1.380649e-23*tmean)
    contributions={k:pref*n*z*z*report['diffusion'][k]['D_1e9_m2_s']*1e-9 for k,n,z in [('FC3',62,3),('FC4',62,4),('K',434,1)]}
    fitq=fit_msd(qmsd,.001,.5,2.)
    report['conductivity']=dict(NE_S_m=float(sum(contributions.values())),NE_contributions_S_m=contributions,
        correlated_mobile_ion_diagnostic_S_m=float(pref*fitq['D_1e9_m2_s']*1e-9),collective_fit=fitq,
        note='NE ignores cross-correlations; collective estimate is diagnostic only for a single short trajectory; fixed neutral-network polarization excluded.')
    report['temperature_K']=tmean
    report['whole_system_COM_drift_nm']=float(np.linalg.norm(drifts[-1])/10)
    report['source_files']={k:dict(bytes=(D/k).stat().st_size,mtime=(D/k).stat().st_mtime) for k in ['prod_10ns.xtc','prod_10ns.tpr','prod_10ns.edr']}
    (O/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    # Small auditable snapshot / setup bundle; no full trajectory transfer.
    u.trajectory[-1]
    u.atoms.write(str(O/'production_end.pdb'))
    mark('analysis complete')

if __name__=='__main__':main()
