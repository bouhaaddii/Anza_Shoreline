"""Create a readable model-architecture figure from the builders in 02_dl_extraction.py."""
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "extraction" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family":"serif","font.size":8.5,"figure.facecolor":"white","savefig.facecolor":"white"})
COLORS={"unet":"#0072B2","segnet":"#E69F00","fcn":"#CC79A7","input":"#56B4E9","shared":"#009E73","ink":"#203044","panel":"#F4F8FA"}
fig,ax=plt.subplots(figsize=(6.65,5.05));ax.set_xlim(0,6.65);ax.set_ylim(0,5.05);ax.axis('off')
# Common input, branching to the three networks.
ib=FancyBboxPatch((1.625,4.48),3.45,.48,boxstyle='round,pad=.03,rounding_size=.06',facecolor='#DDF3FB',edgecolor=COLORS['input'],linewidth=1.2);ax.add_patch(ib)
ax.text(3.325,4.72,"MNDWI patch  •  128 × 128 × 1",ha='center',va='center',fontsize=9,fontweight='bold',color=COLORS['ink'])
models=[
 (0.12,'U-Net','unet',['Encoder levels E1–E4','64 → 128 → 256 → 512','Bottleneck • 1,024 filters','Decoder D4–D1 + skip links','512 → 256 → 128 → 64']),
 (2.25,'SegNet','segnet',['Encoder levels E1–E3','64 → 128 → 256','Upsampling decoder','256 → 128 → 64 • no skip links']),
 (4.38,'FCN','fcn',['Encoder blocks','64 → 128 → 256','Convolution • 512 filters','Bilinear up ×8 • Conv 64']),
]
# Panel width leaves enough room for each label at print size.
for x,name,key,lines in models:
 color=COLORS[key];panel=FancyBboxPatch((x,1.37),2.05,2.42,boxstyle='round,pad=.025,rounding_size=.06',facecolor=COLORS['panel'],edgecolor='#C8D9E3',linewidth=.8);ax.add_patch(panel)
 ax.text(x+1.025,3.62,name,ha='center',va='center',fontsize=11,fontweight='bold',color=color)
 # Input arrows converge from the common MNDWI tile to each network panel.
 ax.add_patch(FancyArrowPatch((3.325,4.46),(x+1.025,3.82),arrowstyle='-|>',mutation_scale=8,linewidth=.85,color=COLORS['input'],connectionstyle='arc3,rad=0'))
 ys=[3.22,2.73,2.24,1.75]
 # Four equal blocks keep the schematic aligned; concise labels describe actual builder stages.
 if key=='unet':
  blocks=[('Encoder E1–E4','64 → 128 → 256 → 512'),('Bottleneck','1,024 filters'),('Decoder D4–D1','512 → 256 → 128 → 64\nSkip connections'),('Output','Sigmoid probability map\n128 × 128')]
 elif key=='segnet':
  blocks=[('Encoder E1–E3','64 → 128 → 256'),('Decoder D3–D1','256 → 128 → 64'),('Upsampling','No skip connections'),('Output','Sigmoid probability map\n128 × 128')]
 else:
  blocks=[('Encoder blocks','64 → 128 → 256'),('Convolution','512 filters'),('Bilinear up ×8','Then convolution • 64 filters'),('Output','Sigmoid probability map\n128 × 128')]
 for i,(head,detail) in enumerate(blocks):
  y=ys[i];box=FancyBboxPatch((x+.12,y-.21),1.81,.42,boxstyle='round,pad=.015,rounding_size=.04',facecolor='white' if i<3 else '#E4F3EC',edgecolor=color if i<3 else COLORS['shared'],linewidth=.85);ax.add_patch(box)
  ax.text(x+1.025,y+.085,head,ha='center',va='center',fontsize=7.6,fontweight='bold',color=COLORS['ink'])
  ax.text(x+1.025,y-.085,detail,ha='center',va='center',fontsize=6.6,color=COLORS['ink'],linespacing=1.0)
  if i<3:ax.add_patch(FancyArrowPatch((x+1.025,y-.21),(x+1.025,ys[i+1]+.21),arrowstyle='-|>',mutation_scale=7,linewidth=.75,color=color))
# Shared downstream processing for every output.
for x,_,_,_ in models:ax.add_patch(FancyArrowPatch((x+1.025,1.34),(3.325,.99),arrowstyle='-|>',mutation_scale=7,linewidth=.8,color=COLORS['shared'],connectionstyle='arc3,rad=0'))
chain=FancyBboxPatch((.22,.23),6.21,.62,boxstyle='round,pad=.03,rounding_size=.06',facecolor='#E8F5EF',edgecolor=COLORS['shared'],linewidth=1.1);ax.add_patch(chain)
ax.text(3.325,.65,"Shared post-processing for all models",ha='center',va='center',fontsize=8.4,fontweight='bold',color=COLORS['ink'])
ax.text(3.325,.43,"0.5 threshold → mask cleaning → marching-squares contour + border filtering → shoreline vector (UTM 29N)",ha='center',va='center',fontsize=7.2,color=COLORS['ink'])
fig.subplots_adjust(left=.01,right=.99,top=.99,bottom=.025)
fig.savefig(OUT/'Fig11_model_architecture.png',dpi=320,bbox_inches='tight',pad_inches=.03)
fig.savefig(OUT/'Fig11_model_architecture.pdf',bbox_inches='tight',pad_inches=.03)
plt.close(fig)
print(OUT/'Fig11_model_architecture.png')
