from json import load
import streamlit as st
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from astropy.table import Table, vstack
from astropy.io import fits
import glob
from utils import ranges
import pickle
from astropy.visualization import ZScaleInterval
import matplotlib as mpl


st.set_page_config(page_title="Forecast", page_icon="👁 ")
st.markdown("#&nbsp; &nbsp; Interactive Visualisation")
st.sidebar.header(" 👁  Visualisation")


@st.cache
def load_cat():
    path = 'data/' 
    cat = []
    # for file in sorted(glob.glob(path+'2sig_new_euc_*flaged.fits'), key=parse_version):
    # for file in sorted(glob.glob(path+'new_euc_*flaged.fits')):

        # print(file)
        # cat.append(Table.read(file))
        # cat[-1]['flag'] = np.zeros_like(cat[-1]['mag'])
    file = "data/cat_1_deep.fits"
    cat = Table.read(file)
    cat['flag'] = np.zeros_like(cat['area'])

    for j in range(len(cat)):
        if cat['blendedness'][j] == 0:
            cat['flag'][j] = -1
        elif (cat['blendedness'][j] < 1) & (cat['blendedness'][j] > 0):
            cat['flag'][j] = 0

        elif (cat['blendedness'][j] > 1)  & (cat['blendedness'][j] < 1000):
            cat['flag'][j] = 1
        else :
            cat['flag'][j] = 2
    return cat

@st.cache()
def load_imgs():
    # segs = np.load('data/new_euc_sim_TU_DC_seg.npy', allow_pickle=True)
    seg = np.load('data/deep_seg.npy', allow_pickle=True)
    # np.save('data/deep_seg.npy', segs[3].array[:3000, :3000])
    # image = fits.open('data/new_euc_sim_TU_DC_WithoutNoise.fits')[0].data
    image = np.load('data/field.npy', allow_pickle=True)
    np.save('data/field.npy', image[:3000, :3000])
    # 2.96e-3, 4.69e-4
    white_noise = np.random.normal(0, 4.69e-4, np.shape(image))
    image_n = image + white_noise[:image.shape[0], :image.shape[1]]
    return (image_n, seg)

@st.cache(hash_funcs={matplotlib.figure.Figure: lambda _: None})
def create_axes():
    fig, ax = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
    return fig, ax


def visual(catalog, fig, img, seg, ax):
    colors = {'-1':'white',
              '0': 'red',
              '1': 'blue',
              '2': 'orange'}
    fig, ax = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
    st.sidebar.markdown("### - Play with the contrast: \n galaxy which don't seem to be blended will appear as so when boosting the contrast to reveal the faint end of the galaxy ")
    st.sidebar.markdown("### - Navigate in the field with the X and Y sliders !")
    # linear  = st.sidebar.checkbox('Linear scale')
    colors = ['white', 'red', 'blue', 'orange']
    cmap = mpl.colors.ListedColormap(colors)

    xs = st.slider('X range', 0, 3000, (754, 1014))
    ys = st.slider('Y range', 0, 3000, (955, 1223))
    contrast = st.slider('Constrast', -1., 1., 0., 0.05)

    img = img[xs[0]:xs[-1], ys[0]:ys[-1]]
    seg = seg[xs[0]:xs[-1], ys[0]:ys[-1]]

    indices = np.where((catalog['X'] < ys[-1]) & (catalog['X'] > ys[0]) & (catalog['Y'] < xs[1]) & (catalog['Y'] > xs[0]))
    new_catalog = catalog[indices]

    a = ax[1].imshow(seg, vmin=0, vmax=3)

    max_ = np.max(img)
    c = ax[0].imshow(img, vmin=np.min(img), vmax=max_ - contrast*max_)

    stars = np.where((np.array(new_catalog['type'])==0))
    ax[1].scatter(np.array(new_catalog['X']).astype('int')-1-ys[0], np.array(new_catalog['Y']).astype('int')-1-xs[0], marker='.', s=10, c=new_catalog['flag'], cmap=cmap, vmin=-1, vmax=2)
    ax[1].legend(loc=(0, 1.2))

    ax[0].set_title('Galaxy field')
    ax[1].set_title('Segmentation map')
    
    ax[0].set_xlabel('Y')
    ax[1].set_xlabel('Y')
    ax[0].set_ylabel('X')
    ax[1].set_ylabel('X')
    st.pyplot(fig)
    fig, ax = plt.subplots(figsize=(10, 0.05))
    ax.scatter([], [], c='blue', marker='.', s=60, label='Blended')
    ax.scatter([], [], c='red', marker='.', s=60, label='Overlapping Only')
    ax.scatter([], [], c='orange', marker='.', s=60, label='Too blended')
    ax.scatter([], [], facecolors='none', linewidth=0.5, edgecolors='black', marker='.', s=100, label='Isolated')
    
    ax.legend(ncol=4)

    ax.axis("off")
    st.pyplot(fig)


fig, ax = create_axes()
catalog = load_cat()
img, seg = load_imgs()
# params  = select_parameters()

visual(catalog, fig, img, seg, ax)
