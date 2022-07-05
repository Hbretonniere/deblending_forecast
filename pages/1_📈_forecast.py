import streamlit as st
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from astropy.table import Table, vstack
import glob
from utils import ranges
import pickle

def load_data():
    with open("data/cats_forecast.pickle", "rb") as input_file:
        cats = pickle.load(input_file)
    return cats

def select_parameters():
    survey = st.sidebar.radio('Select the Survey', ['wide', 'deep'], horizontal=True)
    sigma = st.sidebar.radio('Select the sigma threshold', ['0.8', '1', '2'], horizontal=True)
    seg = f'{sigma} {survey}'
    st.sidebar.markdown('----------------------')
    blend_threshold = st.sidebar.slider('Blendedness Threshold', 0.1, 10., 1., 0.1)
    sup_threshold = st.sidebar.slider('Undectability Threshold', 100, 1000, 500, 10)
    nb_pix = st.sidebar.number_input('Minimum number of pixels for detection', 1, 15, 10)
    st.sidebar.markdown('----------------------')

    type_= st.sidebar.radio('Select the quantity to be plotted', ['percentage', 'Number'], horizontal=True)
    mode = st.sidebar.radio('Select the mask threshold', ['CLEAR', 'Superpose'], horizontal=True)
    if type_ == 'percentage':
        global_max = 100.
    else:
        global_max = 10000
    y_max_1 = st.sidebar.slider('(0, 0) y max', 1., global_max, value=ranges[seg][0])
    y_max_2 = st.sidebar.slider('(0, 1) y max', 1., global_max, value=ranges[seg][1])
    y_max_3 = st.sidebar.slider('(1, 0) y max', 1., global_max, value=ranges[seg][2])
    y_max_4 = st.sidebar.slider('(1, 1) y max', 1., global_max, value=ranges[seg][3])
    return [survey, sigma, blend_threshold, sup_threshold, nb_pix, type_, mode, y_max_1, y_max_2, y_max_3, y_max_4]

# @st.cache(hash_funcs={matplotlib.figure.Figure: lambda _: None}, suppress_st_warning=True)
def forecast(cats, fig, ax, params):
    # st.set_page_config(layout="centered")
    survey, sigma, blend_threshold, sup_threshold, nb_pix, type_, mode, y_max_1, y_max_2, y_max_3, y_max_4 = params
    seg = f'{sigma} {survey}'
    if mode == 'CLEAR':
        ax[0].clear()
        ax[1].clear()
        ax[2].clear()
        ax[3].clear()
        alpha = 0.5
        color = 'cornflowerblue'
    else:
        alpha = 0.1
        color=None
        try:
            y_maxs.append(y_max_1)
        except Exception:
            y_maxs = [y_max_1]
    
    out_cat = cats[seg]
    ax[0].set_axisbelow(True)
    ax[1].set_axisbelow(True)
    ax[2].set_axisbelow(True)
    ax[3].set_axisbelow(True)
    # [axe.set_axisbelow(True) for axe in ax.flatten()]
    if 'wide' in seg:
        sb_lims=[18, 20, 22, 24, 26]

    else:
        sb_lims=[18, 20, 22, 24, 26, 28]

    if mode == "CLEAR":
        ax[0].set_ylim([0, y_max_1])
        ax[1].set_ylim([0, y_max_2])
        ax[2].set_ylim([0, y_max_3])
        ax[3].set_ylim([0, y_max_4])    
    elif (mode == "Superpose")  | (y_max_1 > y_maxs[-1]):
        ax[0].set_ylim([0, y_max_1])
        ax[1].set_ylim([0, y_max_2])
        ax[2].set_ylim([0, y_max_3])
        ax[3].set_ylim([0, y_max_4])   
  
    overlaps = []
    blends = []
    too_blended = []
    detect_blend = []    
    nb_gals = []

    for i in range(len(sb_lims)-1):

            overlap = len(np.where((out_cat['blendedness'] > 0) &
                                   (np.array(out_cat['SB']) > sb_lims[i]) &
                                   (np.array(out_cat['SB']) < sb_lims[i+1])&
                                   (np.array(out_cat['area']) > nb_pix))[0])
            overlaps.append(overlap)

            blend = len(np.where((out_cat['blendedness'] > blend_threshold) &
                                   (np.array(out_cat['SB']) > sb_lims[i])&
                                   (np.array(out_cat['SB']) < sb_lims[i+1])&
                                   (np.array(out_cat['area']) > nb_pix))[0])
            blends.append(blend)
            too_blend = len(np.where((out_cat['blendedness'] > sup_threshold) &
                                   (np.array(out_cat['SB']) > sb_lims[i]) &
                                   (np.array(out_cat['SB']) < sb_lims[i+1])&
                                   (np.array(out_cat['area']) > nb_pix))[0])
            too_blended.append(too_blend)

            nb_gal = len(np.where((np.array(out_cat['SB']) > sb_lims[i]) &
                                   (np.array(out_cat['SB']) < sb_lims[i+1])&
                                   (np.array(out_cat['area']) > nb_pix))[0])
            nb_gals.append(nb_gal)
            detect_blend.append(np.array(blend) - np.array(too_blend))
    
    tot_overlap = np.sum(overlaps)
    tot_blend = np.sum(blends)
    tot_too_blend = np.sum(too_blended)
    tot_detect_blend = np.sum(detect_blend)
    
    if type_ == 'percentage':
        type_ = '%'
        overlaps = np.array(overlaps) / np.array(nb_gals) * 100
        blends = np.array(blends) / np.array(nb_gals)* 100
        detect_blend = detect_blend / np.array(nb_gals) * 100
        too_blended = too_blended / np.array(nb_gals) * 100
        tot_overlap = tot_overlap / np.sum(nb_gals) * 100
        tot_blend = tot_blend / np.sum(nb_gals) * 100
        tot_too_blend = tot_too_blend / np.sum(nb_gals) * 100
        tot_detect_blend = tot_detect_blend / np.sum(nb_gals) * 100
    fs = 15
    

    ax[0].bar(sb_lims[:-1], overlaps, align='edge', width=2, edgecolor='black', color=color, alpha=alpha,  label=seg)

    ax[0].set_xlabel("Surface brightness", fontsize=fs)
    ax[0].set_ylabel(f"{type_} of overlapping galaxies", fontsize=fs)
    ax[0].set_xticks(sb_lims)
    ax[0].set_title(f"Total : {tot_overlap:.1f}%", fontsize=fs)

    ax[1].set_xticks(sb_lims)
    ax[1].bar(sb_lims[:-1], blends, align='edge', width=2, edgecolor='black', color=color, alpha=alpha)#,  label=r'$0.8\sigma$ Deep')
    ax[1].set_xlabel("Surface brightness", fontsize=fs)
    ax[1].set_ylabel(f"{type_} of Blended Galaxy", fontsize=fs)
    ax[1].set_title(f"Total : {tot_blend:.1f}% ", fontsize=fs)

    ax[2].bar(sb_lims[:-1], too_blended, align='edge', width=2, edgecolor='black', alpha=alpha, color=color)#, label=r'$0.8\sigma$ Deep')
    ax[2].set_xlabel("Surface brightness", fontsize=fs)
    ax[2].set_xticks(sb_lims)
    ax[2].set_ylabel(f"{type_} of Undetectable \n Blended Galaxy", fontsize=fs)
    ax[2].set_title(f"Total : {tot_too_blend:.1f}% ", fontsize=fs)

    ax[3].bar(sb_lims[:-1], detect_blend, align='edge', width=2, edgecolor='black', color=color, alpha=alpha)#,  label=r'$0.8\sigma$ Deep')
    ax[3].set_xlabel("Surface brightness", fontsize=fs)
    ax[3].set_ylabel(f"{type_} of Detectable \n Blended Galaxy", fontsize=fs)
    ax[3].set_xticks(sb_lims)
    ax[3].set_title(f"Total : {tot_detect_blend:.1f}%", fontsize=fs)

    if mode == 'Superpose':
        ax[0].legend()

    st.pyplot(fig)
    return 0

st.set_page_config(page_title="Forecast", page_icon="📈")
st.markdown("# 📈 &nbsp; &nbsp; Interactive Blending Forecast")
# st.sidebar.header(" 📈 Forecast")

catalogs = load_data()

@st.cache(hash_funcs={matplotlib.figure.Figure: lambda _: None})
def create_axes():
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    ax = ax.flatten()
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.6, hspace=0.4)
    plt.rc('axes', axisbelow=True)
    return fig, ax

fig, ax = create_axes()

params = select_parameters()
forecast(catalogs, fig, ax, params)
