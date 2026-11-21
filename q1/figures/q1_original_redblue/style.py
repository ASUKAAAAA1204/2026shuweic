"""Figure palette transcribed from the supplied swatches."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = ROOT / 'C_Problem_Data/v1/v1'
RED, PINK, PALERED = '#CE4844', '#F7A9A8', '#FDE6E6'
PALEBLUE, LIGHT, SKY, BLUE, NAVY = '#E8EEF8', '#B6CAED', '#87AEE6', '#2A73BA', '#1F4E89'


def read(file):
    return pd.read_csv(ROOT / 'results/q1_revised' / file)


def start():
    fonts = {font.name for font in font_manager.fontManager.ttflist}
    family = next(f for f in ('Microsoft YaHei','Noto Sans SC','SimHei') if f in fonts)
    plt.rcParams.update({'font.family':family,'font.size':10,'axes.titlesize':12,
                         'axes.titleweight':'bold','axes.titlepad':14,'axes.labelsize':10,
                         'xtick.labelsize':9,'ytick.labelsize':9,'legend.fontsize':8.5,
                         'legend.frameon':False,'axes.spines.top':False,'axes.spines.right':False,
                         'axes.edgecolor':'#BAC3CC','text.color':'#24374D','axes.labelcolor':'#24374D',
                         'axes.unicode_minus':False,'axes.axisbelow':True,'pdf.fonttype':42})


def panel(ax,letter,title):
    ax.set_title(f'{letter}   {title}',loc='left')
