"""Two-page SBMA result review; detailed figures and source tables remain in appendix folder."""
from pathlib import Path
import json
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.utils import ImageReader
from reportlab.platypus import Paragraph
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.colors import HexColor

ROOT=Path(__file__).resolve().parents[2]
B=Path(__file__).resolve().parent
R=json.loads((B/'results/summary.json').read_text())
OUT=ROOT/'output/pdf/sbma_results_20261028'
FIG=OUT/'原始材料/图表'
pdfmetrics.registerFont(TTFont('ArialUnicode','/System/Library/Fonts/Supplemental/Arial Unicode.ttf'))
pdfmetrics.registerFontFamily('ArialUnicode',normal='ArialUnicode',bold='ArialUnicode',italic='ArialUnicode',boldItalic='ArialUnicode')
W,H=A4;M=35
C=canvas.Canvas(str(OUT/'SBMA1.0g_结果核验.pdf'),pagesize=A4)
C.setTitle('SBMA 1.0 g：10 ns结果核验')
STYLE=ParagraphStyle('body',fontName='ArialUnicode',fontSize=10,leading=15,textColor=HexColor('#263B4A'))

def para(text,y,size=10):
    style=ParagraphStyle('local',parent=STYLE,fontSize=size,leading=size*1.5)
    p=Paragraph(text,style);_,h=p.wrap(W-2*M,200);p.drawOn(C,M,y-h);return y-h-7
def title(text,sub,page):
    C.setFillColor(HexColor('#203D54'));C.setFont('ArialUnicode',17);C.drawString(M,H-44,text)
    C.setFont('ArialUnicode',9);C.setFillColor(HexColor('#5F7180'));C.drawString(M,H-63,sub)
    C.setStrokeColor(HexColor('#D8E1E7'));C.line(M,H-74,W-M,H-74)
    C.setFont('ArialUnicode',8);C.drawString(M,22,'单个网络、单条轨迹；时间分块不等于独立重复。');C.drawRightString(W-M,22,str(page)+'/2')
def image(name,y,width=W-2*M):
    p=FIG/(name+'.png');im=ImageReader(str(p));iw,ih=im.getSize();h=width*ih/iw
    C.drawImage(im,M,y-h,width=width,height=h,mask='auto');return y-h-7

title('SBMA 1.0 g｜10 ns 结果核验','2026-10-28  ·  298 K  ·  OPLS + 核心约束RESP + SPC/E',1)
y=H-86
y=para('本步请核验：平衡是否足够；MSD拟合区间是否合理；局部接触阈值及驻留定义是否可用。',y)
y=image('01_运行状态与末态',y)
y=para('100 ps NVT + 5 ns NPT 后，固定体积NVT生产10 ns；步长0.5 fs。网络中青色为SBMA、灰色为AAm、暗红为MBAA。曲线灰色为平衡段，蓝色为生产段；虚线为生产起点。运行正常结束，无LINCS警告。温度、密度稳定不等于网络和离子环境已经充分平衡。',y,9)
y=image('02_扩散与近似电导率',y)
bad=[k for k,v in R['diffusion'].items() if not v['diffusive_regime_screen']]
text='MSD使用1 ps间隔坐标、多时间原点并去除全体系质心漂移；氧化还原离子以Fe中心跟踪。阴影为0.5-2 ns拟合窗；散点为五个2 ns时间块的短窗拟合，用于观察波动。'
text+=' 两氧化态α约为0.75和0.82，尚未清楚达到正常扩散的1；D及NE电导率均按暂定估计报告。'
y=para(text,y,9)
assert y>32, y
C.showPage()

title('局部配位、接触与存留','FC3：Fe(CN)6³⁻（蓝）；FC4：Fe(CN)6⁴⁻（橙）。图中P指SBMA季铵位点。',2)
y=H-86
y=image('03_局部结构与接触动力学',y)
cuts=R['common_cutoffs_nm']
y=para(f"RDF参考原子为CN末端N；邻居分别为水O、K及SBMA季铵N。虚线是两氧化态共用阈值：水{cuts['N3_W']:.3f} nm，K {cuts['N3_K']:.3f} nm，SBMA {cuts['N3_Pplus']:.3f} nm。水和K取两氧化态已分辨首壳层阈值的较大值统一比较。FC4-季铵位点无可分辨首峰，故仅沿用FC3首壳层边界作统一几何邻近统计，不能据此认定FC4存在特异配对。",y,9)
y=para('接触：同一氧化还原离子的任一CN末端N落入阈值即计入；对同一K或同一SBMA位点去重。占有率指至少接触一个邻居的氧化还原离子比例，误差线为五个时间块的标准差。此处统计几何接触，不把更远壳层自动命名为水隔开配对。',y,9)
y=para('存留：按同一离子-邻居或水-壳层配对计数，间隔1 ps，不合并断开后重返。e实线为新生K接触的Kaplan-Meier寿命曲线（计入右删失）；虚线为初始接触的连续剩余存留（单时间原点，不是总寿命）。f为新生水接触曲线。亚ps断裂不能分辨，快速水交换可能受采样间隔影响。',y,9)
y=para('<b>数值摘要（均为本条轨迹估计）</b>',y)
cm=R['contact_means']
y=para(f"K接触占有率：FC3 {cm['F3_K_fraction']*100:.1f}%，FC4 {cm['F4_K_fraction']*100:.1f}%；每个离子的平均接触K数分别为{cm['F3_K_number']:.2f}和{cm['F4_K_number']:.2f}。季铵邻近占有率分别为{cm['F3_P_fraction']*100:.1f}%和{cm['F4_P_fraction']*100:.1f}%。请核验该配对程度是否合理。",y,9)
for names in [('FC3','FC4'),('K','water')]:
    entries=[]
    for name in names:
        d=R['diffusion'][name]
        entries.append(f"{name}: D={d['D_1e9_m2_s']:.4f}, α={d['alpha']:.2f}")
    y=para('；'.join(entries)+'。D单位：10^-9 m²/s。',y,9)
left3=R['lifetimes']['F3_K']['initial_contacts_remaining_fraction']
left4=R['lifetimes']['F4_K']['initial_contacts_remaining_fraction']
y=para(f"初始K接触到10 ns仍连续保持：FC3 {left3*100:.1f}%，FC4 {left4*100:.1f}%。这部分只知道剩余存留至少10 ns，不能给出完整寿命；新生短接触不能代表全部配对。",y,9)
y=para(f"NE电导率估计 {R['conductivity']['NE_S_m']:.3f} S/m，忽略离子间运动关联。移动离子集体电荷位移拟合另存为诊断数据，单条短轨迹不能据此声称已得到收敛的关联电导率。",y,9)
y=para('<b>判断边界：</b>只有SBMA 1.0 g，无独立网络重复，尚不能判断其相对PAM/CBMA的改善；这些量不直接等于器件I-V或响应时间。数据表、逐项曲线、分析脚本和运行证据在“原始材料”中；大轨迹保留在工作站。',y,9)
assert y>32,y
C.save()
print(OUT/'SBMA1.0g_结果核验.pdf')
