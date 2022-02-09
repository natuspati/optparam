import numpy as np
import pandas as pd
from pathlib import Path
from targets import Circles, Checkerboard
from classes import Line, ImageContainer
import img2pdf, sys, os, time
import matplotlib.pyplot as plt
import matlab.engine

def split_fun(original_str, index):
    split_str = original_str.split(',')
    if len(split_str) > 1:
        new_str = f"{split_str[0]}^{index}" + f", {split_str[1]}"
    else:
        new_str = f"{original_str}^{index}"

    return new_str


if __name__ == '__main__':
    eng = matlab.engine.start_matlab()
    eng.quit()

    # plt.close("all")
    # plt.rcParams['font.size'] = '16'
    #
    # # cs = np.loadtxt("csvs\\costs10.csv", delimiter=",")
    # # cs = cs[:,0]
    #
    # for i, el in enumerate(cs):
    #     cs[i] = el/(i+1)

    # xs = list(range(1,len(cs)+1))
    # fig, ax = plt.subplots(figsize=(12, 8))
    # fig.patch.set_facecolor('None')

    # ax.scatter(xs, cs, marker="P", s=120, c="C0")
    # ax.set_ylabel("Cost normalized per image count, mm$^2$")
    # ax.set_xlabel("Number of images, $I$")
    # ax.grid(visible=True, which='both', axis='both')
    # ax.set_xticks(xs)
    # fig.savefig("plots\\evol10cost.png", dpi='figure', format='png', pad_inches=0.0)

    # df = pd.read_csv("csvs\\num_imgs_12_bounded100.csv", delimiter=',',index_col=0)
    #
    # fig, ax = plt.subplots(figsize=(12, 8))
    # fig.patch.set_facecolor('None')
    # xs = list(range(len(df.columns) - 1))
    #
    # syst = df.iloc[:17,1:]
    # a = list(range(12))
    # b = a[2::3]
    # a = set(a)
    # b = set(b)
    # angles = a - b
    # for i in range(17):
    #     if i in angles:
    #         syst.iloc[i] = syst.iloc[i] * 180 / np.pi
    #     elif i in {15,16}:
    #         syst.iloc[i] = syst.iloc[i] * 1000
    #
    #
    # counter=0


    # focal_labels = [r"$focal_{x}$", r"$focal_{y}$", r"$focal_{z}$"]
    # focal_colors = ["C1", "C1", "C1"]
    # focal_markers = ["o", "s", "X"]
    # for i in {12,13,14}:
    #     ax.scatter(xs, syst.iloc[i], label=focal_labels[counter], marker=focal_markers[counter], s=100,
    #                facecolors='none', edgecolors=focal_colors[counter])
    #     counter+=1
    #
    # ax.set_xlabel("Number of images in optimization, $I$")
    # ax.set_ylabel(r"Focal position, mm")
    # ax.set_xticks(list(range(13)))
    # # ax.set_yticks(list(range(-20,30,5)))
    # ax.grid(visible=True, which='both', axis='both')
    #
    # ax2 = ax.twinx()
    # counter = 0
    # scl_labels = [r"$m_{x}$", r"$m_{y}$"]
    # scl_colors = ["C0", "C0"]
    # scl_markers = ["P", "v"]
    # for i in {15,16}:
    #     ax2.scatter(xs, syst.iloc[i], label=scl_labels[counter], marker=scl_markers[counter], s=100,
    #                facecolors='none', edgecolors=scl_colors[counter])
    #     counter+=1
    # ax2.set_ylabel(r"Pixel scaling, $10^{-3}$ mm/px")
    # ax2.legend(loc='upper center', bbox_to_anchor=(0.1, 1.05), ncol=2, fancybox=True, shadow=True)
    # ax2.tick_params(axis='y', labelcolor="C0")
    # ax.tick_params(axis='y', labelcolor="C1")
    # ax.legend(loc='upper center', bbox_to_anchor=(0.75, 1.05), ncol=3, fancybox=True, shadow=True, facecolor='white')
    # fig.savefig("plots\\evol12focal.png", dpi='figure', format='png', pad_inches=0.0)
    
    
    
    
    
    
    
    
    # mdist_labels = [r"$r_{left,in}$", r"$r_{left,ou}$", r"$r_{right,in}$", r"$r_{right,ou}$"]
    # mdist_colors = ["C0", "C0", "C1", "C1"]
    # mdist_markers = ["o", "P", "o", "P"]
    # for i in b:
    #     ax.scatter(xs, syst.iloc[i], label=mdist_labels[counter], marker=mdist_markers[counter], s=100,
    #                facecolors='none', edgecolors=mdist_colors[counter])
    #     counter+=1
    # ax.legend(loc='upper center', bbox_to_anchor=(0.75, 0.9), ncol=2, fancybox=True, shadow=True)
    # ax.set_xlabel("Number of images in optimization, $I$")
    # ax.set_ylabel(r"Mirror position, mm")
    # ax.set_xticks(list(range(13)))
    # ax.set_yticks(list(range(-20,30,5)))
    # ax.grid(visible=True, which='both', axis='both')
    # fig.savefig("plots\\evol12mdists.png", dpi='figure', format='png', pad_inches=0.0)

    # angle_labels = [r"$\theta_{left, in}$", r"$\phi_{left,in}$",r"$\theta_{left, ou}$", r"$\phi_{left,ou}$",
    #                 r"$\theta_{right, in}$", r"$\phi_{right,in}$",r"$\theta_{right, ou}$", r"$\phi_{right,ou}$"]
    # angle_markers = ["o", "s", "P", "X"] * 2
    # angle_colors = ['C0', 'C0', 'b', 'b', 'C1', 'C1', 'orange', 'orange']
    # angle_ecolors = ['face', 'none'] * 4
    # counter= 0
    # for i in angles:
    #     ax.scatter(xs, syst.iloc[i], label=angle_labels[counter], marker=angle_markers[counter], s=100,
    #                facecolors='none', edgecolors=angle_colors[counter])
    #     counter+=1
    # ax.legend(loc='upper center', bbox_to_anchor=(0.8, 0.5), ncol=2, fancybox=True, shadow=True)
    # ax.set_xlabel("Number of images in optimization, $I$")
    # ax.set_ylabel(r"Angle, $^o$")
    # ax.set_xticks(list(range(13)))
    # ax.set_yticks(list(range(0,200,20)))
    # ax.grid(visible=True, which='both', axis='both')
    # fig.savefig("plots\\evol12angles.png", dpi='figure', format='png', pad_inches=0.0)



    # res = np.loadtxt("csvs\\10imgsol.csv", delimiter=',')
    # fig, ax = plt.subplots(figsize=(12, 8))
    # fig.patch.set_facecolor('None')
    # targetinfo = res[17:]
    # Txs = targetinfo[0::6]
    # Tys = targetinfo[1::6]
    # Tzs = targetinfo[2::6]
    # dists = np.zeros_like(Txs)
    # for i, (Tx, Ty, Tz) in enumerate(zip(Txs, Tys, Tzs)):
    #     dists[i] = np.linalg.norm([Tx, Ty, Tz])
    #
    # iTys = np.array([-40, 20, -40, 20, -40, 20, -40, 20, -40, 20])
    # iTxs = np.ones_like(Txs)
    # iTxs = 100 * iTxs
    # iTzs = np.array([270, 270, 290, 290, 310, 310, 330, 330, 350, 350])
    # idists = np.zeros_like(Txs)
    # for i, (Tx, Ty, Tz) in enumerate(zip(iTxs, iTys, iTzs)):
    #     idists[i] = np.linalg.norm([Tx, Ty, Tz])
    #
    # xs = list(range(1, 11))
    # ax.scatter(xs, iTxs, label="$T_x^{meas}$", marker='P', s=100, c='C0')
    # ax.scatter(xs, iTys, label="$T_y^{meas}$", marker='P', s=100, c='C1')
    # ax.scatter(xs, iTzs, label="$T_z^{meas}$", marker='P', s=100, c='C2')
    #
    # # ax.scatter(xs, Txs, label="$T_x^{opt}$", marker='v', s=120, c='b')
    # # ax.scatter(xs, Tys, label="$T_y^{opt}$", marker='v', s=120, c='orange')
    # # ax.scatter(xs, Tzs, label="$T_z^{opt}$", marker='v', s=120, c='g')
    #
    # ax.set_ylabel("Optimized value, mm")
    # ax.set_xlabel("Image index, $I$")
    # ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=3, fancybox=True, shadow=True)
    # ax.grid(visible=True, which='both', axis='both')
    # ax.set_xticks(xs)
    # ytics = list(range(-50, 450, 50))
    # ax.set_yticks(ytics)
    # fig.savefig("plots\\10imgfull.png", dpi='figure', format='png', pad_inches=0.0)
