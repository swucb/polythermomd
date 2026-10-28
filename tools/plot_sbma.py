"""Python-only result figures from the exported single-trajectory source data."""
from pathlib import Path
import json,itertools
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import MDAnalysis as mda

ROOT=Path(__file__).resolve().parents[1]
B=ROOT/'reviews/2026-10-28_SBMA_1.0g/原始材料/脚本与方法'
DATA=ROOT/'reviews/2026-10-28_SBMA_1.0g/原始材料/数据'
OUT=ROOT/'reviews/2026-10-28_SBMA_1.0g/原始材料/图表'
OUT.mkdir(parents=True,exist_ok=True)
R=json.loads((DATA/'summary.json').read_text())
C={'FC3':'#28689D','FC4':'#CF713A','K':'#72846C','water':'#7A99B0'}
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica','DejaVu Sans'],
    'font.size':8,'axes.labelsize':8,'axes.titlesize':9,'xtick.labelsize':7,'ytick.labelsize':7,
    'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,'legend.frameon':False,
    'legend.fontsize':7,'svg.fonttype':'none','pdf.fonttype':42,'savefig.facecolor':'white'})

def read(name):return np.genfromtxt(DATA/(name+'.csv'),delimiter=',',names=True)
def save(fig,name):
    fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight')
    fig.savefig(OUT/(name+'.svg'),bbox_inches='tight')
    fig.savefig(OUT/(name+'.png'),dpi=600,bbox_inches='tight')
    plt.close(fig)
def label(ax,letter,title):ax.set_title(letter+'  '+title,loc='left',fontweight='bold',pad=7)

def snapshot(ax):
    d=ROOT/'reviews/2026-10-28_SBMA_1.0g/原始材料/SBMA_1.0g'
    u=mda.Universe(str(d/'prod_10ns.tpr'),str(d/'prod_10ns.gro'))
    L=u.dimensions[:3]/10;xyz=u.atoms.positions/10;xyz%=L
    amap=json.loads((B/'atom_map_identity.json').read_text())
    yaw=.75;pitch=.45
    Ry=np.array([[np.cos(yaw),0,np.sin(yaw)],[0,1,0],[-np.sin(yaw),0,np.cos(yaw)]])
    Rx=np.array([[1,0,0],[0,np.cos(pitch),-np.sin(pitch)],[0,np.sin(pitch),np.cos(pitch)]])
    rot=Rx@Ry
    def proj(p):return ((p-L/2)@rot.T)[:2]
    segments=[];colors=[]
    palette={'AAM':'#B9C1C8','SBMA':'#388C99','MBAA':'#AE5962'}
    for i,j in u.bonds.indices:
        if u.atoms[i].mass<2 or u.atoms[j].mass<2:continue
        if u.atoms[i].resname=='SOL' or u.atoms[j].resname=='SOL':continue
        p=xyz[i];dv=xyz[j]-p;dv-=L*np.rint(dv/L)
        ts=[0.,1.]
        for k in range(3):
            if abs(dv[k])>1e-10:
                for plane in [0,L[k]]:
                    t=(plane-p[k])/dv[k]
                    if 0<t<1:ts.append(t)
        ts=sorted(set(ts))
        col=palette[amap[i]['kind']] if i<len(amap) else C.get('FC3' if u.atoms[i].resname=='FC3' else 'FC4','#999999')
        for a,b in zip(ts,ts[1:]):
            shift=L*np.floor((p+dv*(a+b)/2)/L)
            segments.append([proj(p+dv*a-shift),proj(p+dv*b-shift)]);colors.append(col)
    ax.add_collection(LineCollection(segments,colors=colors,linewidths=.45,alpha=.8,rasterized=False))
    verts=np.array(list(itertools.product(*[(0,x) for x in L])))
    for i,j in itertools.combinations(range(8),2):
        if np.count_nonzero(verts[i]!=verts[j])==1:
            p=np.array([proj(verts[i]),proj(verts[j])]);ax.plot(p[:,0],p[:,1],color='#B9C1C8',lw=.6,zorder=0)
    for key,res in [('FC3','FC3'),('FC4','FC4')]:
        sel=u.select_atoms('resname '+res+' and name FE')
        p=np.array([proj(x) for x in xyz[sel.indices]])
        ax.scatter(p[:,0],p[:,1],s=9,color=C[key],label=key,zorder=5,edgecolors='white',linewidths=.2)
    ax.autoscale();ax.set_aspect('equal');ax.axis('off');ax.legend(loc='upper right',ncol=2,handletextpad=.2,columnspacing=.6)
    ax.text(.02,.01,f'10 ns endpoint | box {L[0]:.3f} nm\nWater, H and K hidden; periodic bonds clipped.',transform=ax.transAxes,fontsize=7,color='#53616C')

def quality():
    fig=plt.figure(figsize=(7.2,3.55),layout='constrained')
    gs=fig.add_gridspec(3,2,width_ratios=[1.05,1.25])
    ax=fig.add_subplot(gs[:,0]);snapshot(ax);label(ax,'a','SBMA network and redox ions')
    stages=['eq_nvt']+[f'eq_npt_{i:02d}' for i in range(1,6)]+['prod_10ns']
    offset=0;arrays=[]
    for stage in stages:
        a=read(stage+'_energy');t=a['time_ps']/1000; t=t-t[0]+offset
        arrays.append((stage,t,a));offset=t[-1]
    for row,(field,title,ylabel) in enumerate([('temperature_K','Temperature','K'),('density_kg_m3','Density','kg m$^{-3}$'),('potential_kJ_mol','Potential energy','10$^6$ kJ mol$^{-1}$')]):
        ax=fig.add_subplot(gs[row,1]);label(ax,'bcd'[row],title)
        for stage,t,a in arrays:
            y=a[field]/(1e6 if field=='potential_kJ_mol' else 1)
            ax.plot(t,y,color='#28689D' if stage=='prod_10ns' else '#A5AFB8',lw=.35,alpha=.8)
        ax.axvline(5.1,color='#9A6D43',ls='--',lw=.7);ax.set_ylabel(ylabel)
        if row==0:ax.axhline(298,color='#677785',lw=.6,ls=':')
        if row==2:ax.set_xlabel('Elapsed simulation time (ns)')
    save(fig,'01_运行状态与末态')

def transport():
    fig,axes=plt.subplots(2,2,figsize=(7.2,4.25),layout='constrained')
    for name in ['FC3','FC4','K']:
        a=read('msd_'+name);axes[0,0].plot(a['lag_ns'],a['MSD_nm2'],color=C[name],label=name,lw=1.2)
        t=np.linspace(.5,2,60);fit=R['diffusion'][name]
        axes[0,0].plot(t,6*fit['D_1e9_m2_s']*t+fit['intercept_nm2'],color=C[name],ls='--',lw=1)
    ax=axes[0,0];label(ax,'a','Ion MSD');ax.axvspan(.5,2,color='#EAEFF2',zorder=-2);ax.set(xlabel='Lag time (ns)',ylabel='MSD (nm$^2$)');ax.legend(ncol=3)
    a=read('msd_water');ax=axes[0,1];ax.plot(a['lag_ns'],a['MSD_nm2'],color=C['water']);ax.axvspan(.5,2,color='#EAEFF2',zorder=-2);label(ax,'b','Water MSD');ax.set(xlabel='Lag time (ns)',ylabel='MSD (nm$^2$)')
    ax=axes[1,0];names=['FC3','FC4','K','water'];ys=[R['diffusion'][n]['D_1e9_m2_s'] for n in names]
    ax.bar(names,ys,color=[C[n] for n in names],width=.6)
    # Block estimates use a shorter lag window and are displayed as individual points, not CI bars.
    for i,n in enumerate(names):ax.scatter(i+np.linspace(-.15,.15,5),R['diffusion'][n]['block_D_1e9_m2_s'],s=10,c='#273846',marker='.',zorder=4)
    label(ax,'c','Apparent diffusion coefficients');ax.set_ylabel('$D$ (10$^{-9}$ m$^2$ s$^{-1}$)')
    ax.text(.02,.98,'Bars: 0.5-2 ns fit\nDots: five 2 ns blocks, 0.2-0.8 ns fit',transform=ax.transAxes,va='top',fontsize=6.6)
    ax=axes[1,1];ys=[R['conductivity']['NE_contributions_S_m'][k] for k in ['FC3','FC4','K']]
    ax.bar(['FC3','FC4','K'],ys,color=[C[k] for k in ['FC3','FC4','K']],width=.6);label(ax,'d','Nernst-Einstein estimate');ax.set_ylabel('Contribution (S m$^{-1}$)')
    ax.set_ylim(0,max(ys)*1.4)
    ax.text(.02,.98,f"Total {R['conductivity']['NE_S_m']:.2f} S m$^{{-1}}$\nIon correlations omitted",transform=ax.transAxes,ha='left',va='top',fontsize=7)
    save(fig,'02_扩散与近似电导率')

def local():
    fig,axes=plt.subplots(2,3,figsize=(7.2,4.5),layout='constrained')
    for col,(suffix,title) in enumerate([('W','CN nitrogen - water O'),('K','CN nitrogen - K'),('Pplus','CN nitrogen - SBMA N$^+$')]):
        ax=axes[0,col]
        for key,name in [('N3','FC3'),('N4','FC4')]:
            a=read('rdf_'+key+'_'+suffix);ax.plot(a['r_nm'],a['g_r'],color=C[name],label=name)
        ax.axvline(R['common_cutoffs_nm']['N3_'+suffix],color='#71808D',ls='--',lw=.7)
        label(ax,'abc'[col],title);ax.set(xlabel='Distance (nm)',ylabel='$g(r)$',xlim=(.15,1.05));ax.legend()
    ax=axes[1,0];cols=['F3_K_fraction','F4_K_fraction','F3_P_fraction','F4_P_fraction'];ys=[R['contact_means'][x]*100 for x in cols];err=[R['contact_block_sd'][x]*100 for x in cols]
    ax.bar(np.arange(4),ys,yerr=err,color=[C['FC3'],C['FC4']]*2,capsize=2,width=.65);ax.set_xticks(np.arange(4),['FC3\nK','FC4\nK','FC3\nSBMA','FC4\nSBMA']);ax.set_ylim(0,110);ax.set_ylabel('Redox ions with contact (%)');label(ax,'d','Contact occupancy')
    ax=axes[1,1]
    for state,color in [('F3',C['FC3']),('F4',C['FC4'])]:
        a=read('survival_'+state+'_K');ax.plot(a['time_ps'],a['continuous_survival_KM'],color=color,label=state.replace('F','FC')+' new',lw=1)
        a=read('initial_cohort_'+state+'_K');ax.plot(a['time_ps'],a['continuous_remaining_survival'],color=color,ls='--',label=state.replace('F','FC')+' initial',lw=1)
    ax.set(xscale='log',xlim=(1,10000),ylim=(0,1.03),xlabel='Continuous contact time (ps)',ylabel='Survival probability');label(ax,'e','K contact survival');ax.legend(fontsize=6,ncol=1,loc='upper left',bbox_to_anchor=(.52,.65),borderaxespad=0)
    ax=axes[1,2]
    for name,col in [('F3',C['FC3']),('F4',C['FC4']),('Pplus','#658079'),('SO','#9B7BA3')]:
        a=read('survival_water_'+name);ax.plot(a['time_ps'],a['continuous_survival_KM'],color=col,label={'F3':'FC3 shell','F4':'FC4 shell','Pplus':'SBMA N$^+$','SO':'SBMA SO$_3^-$'}[name],lw=1)
    ax.set(xscale='log',xlim=(1,1000),ylim=(0,1.03),xlabel='Continuous contact time (ps)',ylabel='Survival probability');label(ax,'f','Water contact survival');ax.legend(fontsize=6)
    save(fig,'03_局部结构与接触动力学')

if __name__=='__main__':quality();transport();local()
