import os
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
import sys
sys.path.append('../utils')

import numpy as np
import matplotlib.pyplot as plt
import torch

from load_data import *
from training import *
from evaluate import *
from visualize import *
from cnn_model import *

DIR = "../figures/"
EXT = ".eps"
DPI = 300

def to_path(name):
    return DIR + name + EXT

def get_datasets(dset):
    scale = 1 if dset=="temp" else 10000
    datasets_vor = load_tr_te_od_data(f"../data/{dset}_vor_w.mat", f"../data/{dset}_vor_o.mat", scale=scale)
    datasets_lat = load_tr_te_od_data(f"../data/{dset}_lat_w.mat", f"../data/{dset}_lat_o.mat", scale=scale)
    datasets = dict()
    for key in datasets_vor:
        datasets[key] = datasets_vor[key] + datasets_lat[key]
    return datasets_vor, datasets_lat, datasets

def plot_stress_visualizations(output_file="visualize-stress"):
    output_file = to_path(output_file)
    print(f"Creating stress visualization figure: {output_file}")
    if os.path.exists(output_file):
        print("Figure already exists. Skipping.")
        return

    model = torch.load("../models/multi_model_6.pth")
    _, _, datasets = get_datasets("stress")
    vals = eval_model_all(model, datasets["te"])
    order = np.argsort(vals)
    N = len(vals)
    # Near-median ranks hard-coded to give 1 voronoi & 1 lattice:
    ranks = [-1, N//2 + 2, N//2, 0] 
    plot_comparison(model, [datasets["te"][order[rank]] for rank in ranks], filename=output_file, dpi=DPI)

def plot_temperature_visualizations(output_file="visualize-temperature"):
    output_file = to_path(output_file)
    print(f"Creating temperature visualization figure: {output_file}")
    if os.path.exists(output_file):
        print("Figure already exists. Skipping.")
        return

    model = torch.load("../models/temp_multi_model_6.pth")
    _, _, datasets = get_datasets("temp")
    vals = eval_model_all(model, datasets["te"])
    order = np.argsort(vals)
    N = len(vals)
    # Near-median ranks hard-coded to give 1 voronoi & 1 lattice:
    ranks = [N//2 + 2, N//2] # [-1, N//2 + 2, N//2, 0] 
    plot_comparison(model, [datasets["te"][order[rank]] for rank in ranks], filename=output_file, dpi=DPI)

def plot_r2_figures():
    if os.path.exists(to_path("box")) and os.path.exists(to_path("violin")) and os.path.exists(to_path("parametric-study")):
        print("Skipping box plot, violin plot, and parametric study plot")
        return
    print("Creating box plot, violin plot, and parametric study plot")
    _, _, datasets = get_datasets("stress")
    layer_counts = np.arange(1, 6 + 1)
    r2s = dict(tr=[],te=[],od=[])
    for i in layer_counts:
        model = torch.load(f"../models/multi_model_{i}.pth")
        vals = eval_model_multiple(model, datasets)
        for key in r2s:
            r2s[key].append(np.median(vals[key]))

    # vals now contains r2s for 6-layer network
    plot_boxes(vals, filename=to_path("box"))
    plot_violins(vals, filename=to_path("violin"))

    plt.figure(figsize=[7,5], dpi=DPI)
    plt.title("Model Performance by Layer Count")
    plt.plot(layer_counts, r2s["tr"], "o-", label="Training")
    plt.plot(layer_counts, r2s["te"], "o-", label="Testing")
    plt.plot(layer_counts, r2s["od"], "o-", label="Out-of-Distribution")

    plt.xlabel("Layers")
    plt.ylabel("Median $R^2$")
    plt.legend(loc="lower right")
    plt.savefig(to_path("parametric-study"), bbox_inches="tight")

def plot_high_resolution(output_file="visualize-coarse-fine"):
    output_file = to_path(output_file)
    print(f"Creating high-resolution visualization figure: {output_file}")
    if os.path.exists(output_file):
        print("Figure already exists. Skipping.")
        return

    coarse_data = load_matlab_dataset("../data/stress_vor_w.mat")
    fine_data = load_matlab_dataset("../data/stress_vor_fine.mat")
    model = torch.load("../models/multi_model_6.pth")

    i = 1
    plot_comparison(model, [coarse_data[i], fine_data[i]], filename=output_file, dpi=DPI)

def plot_loss(output_file="loss"):
    loss_tr = np.array([0.007478375725440856, 0.0036990608396990864, 0.0030577165752492873, 0.002893851950352655, 0.0022781846450106967, 0.0020780296835118863, 0.0018717199066213652, 0.0016131077094678403, 0.0016363522035248933, 0.001425421256149093, 0.0014213686468838205, 0.001216722708963971, 0.0012814322481949603, 0.001222603268788589, 0.0011127297167536198, 0.0010715291013593741, 0.0011328943563103167, 0.00106776542030957, 0.0010101935215379853, 0.0008958314606206841, 0.00102021849318362, 0.0009324140079343124, 0.0008990277164139115, 0.0009686599842461873, 0.0008145781969324162, 0.000853372618114463, 0.0008210432499663511, 0.0008622476046298289, 0.0007920665495771572, 0.0007974277735911528, 0.0007755105816386276, 0.0007311950559369506, 0.000781549416337839, 0.000751345327701074, 0.0006823869087997991, 0.0007328272589052176, 0.0006983892765765631, 0.0006839762956747108, 0.0006728439800212982, 0.000647607604436189, 0.0006590195863009285, 0.0006641869651343768, 0.0006605455084218192, 0.0006726071546904678, 0.0006077460414894631, 0.0006629650811532884, 0.0005864614238953437, 0.0006578215706815626, 0.0005741744494798696, 0.0006038477199604131])
    loss_val = np.array([0.007214804731147524, 0.0038696097687534346, 0.00321065676964281, 0.0030778005567435682, 0.002494943418009825, 0.00216112320445518, 0.002095321982214955, 0.0017211405410807856, 0.0018430974002649236, 0.0016250860308969096, 0.001642669530619969, 0.0014250938286977544, 0.0014918187377429603, 0.0014454874777152327, 0.0013504709784524494, 0.0013187198644754971, 0.001365897264272462, 0.0013370073507348935, 0.0012577711517906208, 0.0011953209169678304, 0.0013341652675353543, 0.001221328171955065, 0.0011899198782452914, 0.0012096930256757332, 0.0010971983304716559, 0.0011776704525937021, 0.0011089432394464894, 0.0011332218410598216, 0.0011173855482002181, 0.001098833341923182, 0.0011017843258355241, 0.0010332701637321407, 0.0010805943000650586, 0.0010421149225339833, 0.001010948743262361, 0.0010711088910170473, 0.001043553985091421, 0.0010572762506069467, 0.0010080382721685056, 0.0010121125933369513, 0.0010194134773246332, 0.0010156254511161933, 0.001008461716767215, 0.0009990572034030264, 0.0009585910933276409, 0.0010312796247899313, 0.0009353335302375854, 0.0009959211974728533, 0.0009567242570778944, 0.0009450282613806849])
    
    loss_tr_vor = np.array([0.01284800483539584, 0.005988379871378129, 0.004999395550330518, 0.004033567110127478, 0.003443383445010113, 0.003250558305771847, 0.002705981954059098, 0.002599201399680169, 0.0025998864181747193, 0.002172925937520631, 0.002043955442222796, 0.0022816293587675316, 0.00206610183329758, 0.0023825616682734106, 0.0017959544161931262, 0.0017142413562760339, 0.0015830997842385842, 0.0015875409346153902, 0.001681976078052685, 0.0015037017171016488, 0.0014680870682423118, 0.0014001009353523841, 0.0014873801543944865, 0.001311106092780392, 0.0012899258187280794, 0.0013173951971839416, 0.0013139994992525316, 0.001264768267301406, 0.0012736182301159714, 0.0011541000796023583, 0.0012331895283386985, 0.001111633736618387, 0.0011634565304848366, 0.0010792007696727524, 0.0010899040970616626, 0.0010443601087354183, 0.001039942795268871, 0.0009394016126316274, 0.001011410785249609, 0.0010277819220391392, 0.0010046841658368066, 0.001028457320862799, 0.0009755499018137925, 0.0008504892491328065, 0.0011447803950068191, 0.0008288058664402343, 0.0009326873939244252, 0.0009819450095528736, 0.0008343402672289813, 0.0008446842042758362])
    loss_val_vor = np.array([0.013123542910834658, 0.006145780808656127, 0.005274525105560315, 0.004272302086355921, 0.0038218177237286, 0.0035408509214539664, 0.0030785156289857697, 0.00292877664356638, 0.002976872799335979, 0.0025739779573450503, 0.00243654957963372, 0.0026225769800294076, 0.0025303698084462667, 0.002776986937842594, 0.0022230288115315487, 0.002112802809842833, 0.0020168431259298815, 0.002060048705789086, 0.002286473959848081, 0.0020552467697416434, 0.001968200483752298, 0.001902551738821785, 0.0020090840748525807, 0.0018314060369812068, 0.0018420294184579688, 0.0018763501963076122, 0.001869580939583102, 0.001819255112677638, 0.0018677282218050096, 0.0017483202882431215, 0.0018498856577025436, 0.0016741507116421416, 0.00175999808279812, 0.0016571511071015266, 0.0016647675724198053, 0.0016768461188985384, 0.001615731320471241, 0.001552695569334901, 0.0016803843777233852, 0.0016750362553466403, 0.0016350806531227136, 0.0016423698524886277, 0.001618987008369004, 0.0014992726474338269, 0.001779546020206908, 0.0014990955401299288, 0.0015859442399778344, 0.0016126035580782626, 0.0015162810062247444, 0.001509729734461871])

    loss_tr_lat = np.array([0.0006323708923355298, 0.0002843121134537796, 0.00021348757165469577, 0.00018484615529587245, 0.0001768351589021222, 0.00015598855200551042, 0.00014834423973297817, 0.00014040220188462626, 0.00012425318639543547, 0.00013203518273712688, 0.00011569493138836152, 0.00011422127942296356, 0.00010387685285081716, 0.00010318805116867225, 0.0001083656372452424, 9.425459243175282e-05, 0.00011280251340281211, 9.119221898572505e-05, 9.145734856474518e-05, 9.432752281441026e-05, 8.413560695089473e-05, 8.527432257096734e-05, 8.052109359141469e-05, 8.1365869906449e-05, 7.524678517484062e-05, 8.574416564215426e-05, 7.623113961471972e-05, 7.485495299306422e-05, 6.916934052583201e-05, 8.104943169087164e-05, 7.424753082659664e-05, 6.945931510301761e-05, 6.833329912524278e-05, 6.667454239334347e-05, 6.911305974426795e-05, 7.281032580067404e-05, 6.2690842545976e-05, 6.677297689293482e-05, 6.422868976187602e-05, 6.895740464017308e-05, 5.9602380321734924e-05, 6.344967911104505e-05, 6.0548371415052316e-05, 5.848392917641831e-05, 6.533181387794684e-05, 5.7282021118112424e-05, 6.0574936175612495e-05, 6.252771342929008e-05, 5.911906845199155e-05, 6.073241021624654e-05])
    loss_val_lat = np.array([0.0006313961454816309, 0.00028412948265668094, 0.0002171563754473027, 0.00019359235632009585, 0.00018377447518560074, 0.000165920311019363, 0.00015617038259051696, 0.00014846646743990277, 0.00013249318478074202, 0.000142528596700231, 0.0001248693631430342, 0.00012404880713575038, 0.00011733854256704035, 0.00011448817869563754, 0.00011992178595619407, 0.00010582840112306257, 0.00012525013313279486, 0.00010166783986051086, 0.00010755641599189403, 0.00010428287964828087, 9.775728082104252e-05, 9.830167719428573e-05, 9.300316331746216e-05, 9.445704997915528e-05, 8.696215467011825e-05, 9.995268963280069e-05, 8.831525418997899e-05, 8.677196820826794e-05, 8.217300202659317e-05, 9.27334028142468e-05, 8.5190320496622e-05, 8.235876543039921e-05, 8.209738153084346e-05, 8.03610532238963e-05, 8.108690192443647e-05, 8.574232438718354e-05, 7.73019395336405e-05, 8.067058065876154e-05, 7.7776817320796e-05, 8.437532250354707e-05, 7.397593886366849e-05, 7.805061454860151e-05, 7.675721659666123e-05, 7.281434450419511e-05, 7.921065040363828e-05, 7.08909747834241e-05, 7.411623239022446e-05, 7.82469192949975e-05, 7.349676443880071e-05, 7.784867776081228e-05])

    plt.figure(figsize=[6, 4], dpi=DPI)


    plt.plot(np.arange(len(loss_tr))+1, loss_tr, "-", color="indigo", label="Training, Combined", zorder=9)
    plt.plot(np.arange(len(loss_tr))+1, loss_val, "--", color="indigo", label="Validation, Combined", zorder=9)

    plt.plot(np.arange(len(loss_tr))+1, loss_tr_vor, "-", color="lightcoral", label="Training, Voronoi")
    plt.plot(np.arange(len(loss_tr))+1, loss_val_vor, "--", color="lightcoral", label="Validation, Voronoi")

    plt.plot(np.arange(len(loss_tr))+1, loss_tr_lat, "-", color="royalblue", label="Training, Lattice")
    plt.plot(np.arange(len(loss_tr))+1, loss_val_lat, "--", color="royalblue", label="Validation, Lattice")

    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")

    plt.legend()
    plt.savefig(to_path(output_file), bbox_inches="tight")
    plt.close()



if __name__ == "__main__":
    plot_stress_visualizations()
    plot_temperature_visualizations()
    plot_r2_figures()
    plot_high_resolution()
    plot_loss()