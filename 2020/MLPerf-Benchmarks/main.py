import pandas as pd
import streamlit as st
import plotly.express as px
from PIL import Image


@st.cache
def load_data(file):
    data = pd.read_csv(file)
    return data


#####################
### HTML SETTINGS ###
#####################

ucloud_color = '#006AFF'
pwrai_color = 'red'
dgx1_color = 'green'
dgx2_color = 'gold'
body_color = '#F5FFFA'
header_color = 'black'
subheader_color = '#c00'
code_color = '#c00'
plt_bkg_color = body_color

header = '<style>h1{color: %s;}</style>' % (header_color)
subheader = '<style>h2{color: %s;}</style>' % (subheader_color)
body = '<style>body{background-color: %s;}</style>' % (body_color)
code = '<style>code{color: %s; }</style>' % (code_color)

sidebar = """
  <style>
    # .reportview-container {
    #   flex-direction: row-reverse;
    # }

    # header > .toolbar {
    #   flex-direction: row-reverse;
    #   left: 1rem;
    #   right: auto;
    # }

    # .sidebar .sidebar-collapse-control,
    # .sidebar.--collapsed .sidebar-collapse-control {
    #   left: auto;
    #   right: 0.5rem;
    # }
   
    .sidebar .sidebar-content {
      transition: margin-right .3s, box-shadow .3s;
      background-image: linear-gradient(180deg,%s,%s);
      width: 20rem;
    }
   
    # .sidebar.--collapsed .sidebar-content {
    #   margin-left: auto;
    #   margin-right: -20rem;
    # }

    @media (max-width: 991.98px) {
      .sidebar .sidebar-content {
        margin-left: auto;
      }
    }
  </style>
""" % (ucloud_color, body_color)

st.markdown(header, unsafe_allow_html=True)
st.markdown(subheader, unsafe_allow_html=True)
st.markdown(body, unsafe_allow_html=True)
st.markdown(code, unsafe_allow_html=True)
st.markdown(sidebar, unsafe_allow_html=True)

#############
### TITLE ###
#############

"""
# UCloud Report

## Deep Learning Benchmark Tests
"""

st.write("-------")

###################
### DESCRIPTION ###
###################

description = """
### Last update:
July 31, 2020

### Author:
Emiliano Molinaro (<molinaro@imada.sdu.dk>)\n
Computational Scientist \n
eScience Center \n
Syddansk Universitet

### Desciption:

This report summarizes the results of _single-node data-parallel training_ tests done on the UCloud interactive HPC 
platform and the IBM PowerAI system, based on [MLPerf training benchmarks](https://mlperf.org/training-overview/#overview). 
Each benchmark measures the wallclock time required to train a model on the specified dataset to achieve the 
specified quality target. 

The tests are done using the NVIDIA CUDA-X software stack running on NVIDIA Volta GPUs. The latter leverage the built-in 
NVIDIA [Tensor Cores](https://www.nvidia.com/en-us/data-center/tensor-cores/) technology to accelerate single- and 
[mixed-precision](https://developer.nvidia.com/automatic-mixed-precision) computing. 

The results are compared with the performance of NVIDIA DGX-1/DGX-2 systems reported 
[here](https://github.com/NVIDIA/DeepLearningExamples).

### Specs:

The runtime system used on UCloud is the `u1-gpu-4` machine type: 
- 4 NVIDIA Volta GPUs 
- 78 CPU cores
- 185 GB of memory

"""

#################
### SIDE MENU ###
#################

# logo = Image.open('figs/logo_esc.png')
# st.sidebar.image(logo, format='PNG', width=50)
st.sidebar.title("Benchmark Models")

radio = st.sidebar.radio(label="", options=["Description",
                                            "Benchmark 1",
                                            "Benchmark 2",
                                            "Benchmark 3",
                                            "Benchmark 4",
                                            "Benchmark 5"])

if radio == "Benchmark 1":

    ###############################
    st.subheader("**Benchmark 1**")
    ###############################

    st.markdown("""
        **Category:** 
        Recommender Systems

        **Model:**
        [Neural Collaborative Filtering](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Recommendation/NCF)

        **Framework:**
        PyTorch

        """)

    path_to_file = "training/PyTorch/Recommendation/NCF/results.csv"

    df1 = load_data(path_to_file)
    cols1 = list(df1.columns.values)

    if st.checkbox("Show data benchmark 1"):
        st.table(df1)

    dfp1 = df1.loc[:, [cols1[0], cols1[1], cols1[4], cols1[6]]]
    dfp1 = dfp1.rename(columns={cols1[4]: 'Time to train (s)'})
    dfp1 = dfp1.rename(columns={cols1[6]: 'Throughput (samples/s)'})
    dfp1['Training type'] = 'FP32'

    dfp2 = df1.loc[:, [cols1[0], cols1[1], cols1[5], cols1[7]]]
    dfp2 = dfp2.rename(columns={cols1[5]: 'Time to train (s)'})
    dfp2 = dfp2.rename(columns={cols1[7]: 'Throughput (samples/s)'})
    dfp2['Training type'] = 'Mixed precision'

    dff = pd.concat([dfp1, dfp2])

    # st.table(dff)

    cols = list(dff.columns.values)

    fig1 = px.bar(dff,
                  y=cols[3],
                  x=cols[1],
                  color=cols[0],
                  barmode="group",
                  color_discrete_map={'UCloud': ucloud_color,
                                      'IBM PowerAI': pwrai_color,
                                      'NVIDIA DGX-1': dgx1_color,
                                      'NVIDIA DGX-2': dgx2_color
                                      },
                  hover_data=[cols[1], cols[3]],
                  facet_col=cols[4]
                  )

    fig1.update_xaxes(tickvals=[1, 2, 3, 4, 8, 16], title_text="Number of GPUs", linecolor='black')
    fig1.update_yaxes(linecolor='black', showgrid=True, gridwidth=1, gridcolor='LightGrey')
    fig1.update_layout({'paper_bgcolor': plt_bkg_color, 'plot_bgcolor': plt_bkg_color})

    fig2 = px.bar(dff,
                  y=cols[2],
                  x=cols[1],
                  color=cols[0],
                  barmode="group",
                  color_discrete_map={'UCloud': ucloud_color,
                                      'IBM PowerAI': pwrai_color,
                                      'NVIDIA DGX-1': dgx1_color,
                                      'NVIDIA DGX-2': dgx2_color
                                      },
                  hover_data=[cols[1], cols[2]],
                  facet_col=cols[4]
                  )

    fig2.update_xaxes(tickvals=[1, 2, 3, 4, 8, 16], title_text="Number of GPUs", linecolor='black')
    fig2.update_yaxes(linecolor='black', showgrid=True, gridwidth=1, gridcolor='LightGrey')
    fig2.update_layout({'paper_bgcolor': plt_bkg_color, 'plot_bgcolor': plt_bkg_color})

    st.plotly_chart(fig1)
    st.plotly_chart(fig2)

elif radio == "Benchmark 2":

    ###############################
    st.subheader("**Benchmark 2**")
    ###############################

    st.markdown("""
    **Category:** 
    Computer Vision

    **Model:**
    [SSD300 v1.1](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Detection/SSD)

    **Framework:**
    PyTorch

    """)

    path_to_file = "training/PyTorch/Detection/SSD/results.csv"

    df2 = load_data(path_to_file)
    cols2 = list(df2.columns.values)

    if st.checkbox("Show data benchmark 2"):
        st.table(df2)

    dfp1 = df2.loc[:, [cols2[0], cols2[1], cols2[3], cols2[5]]]
    dfp1 = dfp1.rename(columns={cols2[3]: 'Time to train (s)'})
    dfp1 = dfp1.rename(columns={cols2[5]: 'Throughput (images/s)'})
    dfp1['Training type'] = 'FP32'

    dfp2 = df2.loc[:, [cols2[0], cols2[1], cols2[4], cols2[6]]]
    dfp2 = dfp2.rename(columns={cols2[4]: 'Time to train (s)'})
    dfp2 = dfp2.rename(columns={cols2[6]: 'Throughput (images/s)'})
    dfp2['Training type'] = 'Mixed precision'

    dff = pd.concat([dfp1, dfp2])

    # st.table(dff)

    cols = list(dff.columns.values)

    fig1 = px.bar(dff,
                  y=cols[3],
                  x=cols[1],
                  color=cols[0],
                  barmode="group",
                  color_discrete_map={'UCloud': ucloud_color,
                                      'IBM PowerAI': pwrai_color,
                                      'NVIDIA DGX-1': dgx1_color,
                                      },
                  hover_data=[cols[1], cols[3]],
                  facet_col=cols[4]
                  )

    fig1.update_xaxes(tickvals=[1, 2, 4, 8], title_text="Number of GPUs", linecolor='black')
    fig1.update_yaxes(linecolor='black', showgrid=True, gridwidth=1, gridcolor='LightGrey')
    fig1.update_layout({'paper_bgcolor': plt_bkg_color, 'plot_bgcolor': plt_bkg_color})

    fig2 = px.bar(dff,
                  y=cols[2],
                  x=cols[1],
                  color=cols[0],
                  barmode="group",
                  color_discrete_map={'UCloud': ucloud_color,
                                      'IBM PowerAI': pwrai_color,
                                      'NVIDIA DGX-1': dgx1_color,
                                      },
                  hover_data=[cols[1], cols[2]],
                  orientation='v',
                  facet_col=cols[4]
                  )

    fig2.update_xaxes(tickvals=[1, 2, 4, 8], title_text="Number of GPUs", linecolor='black')
    fig2.update_yaxes(linecolor='black', showgrid=True, gridwidth=1, gridcolor='LightGrey')
    fig2.update_layout({'paper_bgcolor': plt_bkg_color, 'plot_bgcolor': plt_bkg_color})

    st.plotly_chart(fig1)
    st.plotly_chart(fig2)

elif radio == "Benchmark 3":

    ###############################
    st.subheader("**Benchmark 3**")
    ###############################

    st.markdown("""
    **Category:** 
    Natural Language Processing

    **Model:**
    [GNMT v2](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Translation/GNMT)

    **Framework:**
    PyTorch

    """)

    path_to_file = "training/PyTorch/Translation/GNMT/results.csv"

    df3 = load_data(path_to_file)
    cols3 = list(df3.columns.values)

    if st.checkbox("Show data benchmark 3"):
        st.table(df3)

    dfp1 = df3.loc[:, [cols3[0], cols3[1], cols3[5], cols3[7]]]
    dfp1 = dfp1.rename(columns={cols3[5]: 'Time to train (min)'})
    dfp1 = dfp1.rename(columns={cols3[7]: 'Throughput (tok/s)'})
    dfp1['Training type'] = 'FP32'

    dfp2 = df3.loc[:, [cols3[0], cols3[1], cols3[6], cols3[8]]]
    dfp2 = dfp2.rename(columns={cols3[6]: 'Time to train (min)'})
    dfp2 = dfp2.rename(columns={cols3[8]: 'Throughput (tok/s)'})
    dfp2['Training type'] = 'Mixed precision'

    dff = pd.concat([dfp1, dfp2])

    # st.table(dff)

    cols = list(dff.columns.values)

    fig1 = px.bar(dff,
                  y=cols[3],
                  x=cols[1],
                  color=cols[0],
                  barmode="group",
                  color_discrete_map={'UCloud': ucloud_color,
                                      'IBM PowerAI': pwrai_color,
                                      'NVIDIA DGX-1': dgx1_color,
                                      'NVIDIA DGX-2': dgx2_color
                                      },
                  hover_data=[cols[1], cols[3]],
                  facet_col=cols[4]
                  )

    fig1.update_xaxes(tickvals=[1, 2, 4, 8, 16], title_text="Number of GPUs", linecolor='black')
    fig1.update_yaxes(linecolor='black', showgrid=True, gridwidth=1, gridcolor='LightGrey')
    fig1.update_layout({'paper_bgcolor': plt_bkg_color, 'plot_bgcolor': plt_bkg_color})

    fig2 = px.bar(dff,
                  y=cols[2],
                  x=cols[1],
                  color=cols[0],
                  barmode="group",
                  color_discrete_map={'UCloud': ucloud_color,
                                      'IBM PowerAI': pwrai_color,
                                      'NVIDIA DGX-1': dgx1_color,
                                      'NVIDIA DGX-2': dgx2_color
                                      },
                  hover_data=[cols[1], cols[2]],
                  facet_col=cols[4]
                  )

    fig2.update_xaxes(tickvals=[1, 2, 4, 8, 16], title_text="Number of GPUs", linecolor='black')
    fig2.update_yaxes(linecolor='black', showgrid=True, gridwidth=1, gridcolor='LightGrey')
    fig2.update_layout({'paper_bgcolor': plt_bkg_color, 'plot_bgcolor': plt_bkg_color})

    st.plotly_chart(fig1)
    st.plotly_chart(fig2)

elif radio == "Benchmark 4":

    ###############################
    st.subheader("**Benchmark 4**")
    ###############################

    st.markdown("""
    **Category:** 
    Speech Synthesis

    **Model:**
    [Tacotron 2](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechSynthesis/Tacotron2)

    **Framework:**
    PyTorch

    """)

    path_to_file = "training/PyTorch/SpeechSynthesis/Tacotron2/results.csv"

    df4 = load_data(path_to_file)
    cols4 = list(df4.columns.values)

    if st.checkbox("Show data benchmark 4"):
        st.table(df4)

    dfp1 = df4.loc[:, [cols4[0], cols4[1], cols4[4], cols4[6]]]
    dfp1 = dfp1.rename(columns={cols4[4]: 'Time to train (h)'})
    dfp1 = dfp1.rename(columns={cols4[6]: 'Throughput (mels/s)'})
    dfp1['Training type'] = 'FP32'

    dfp2 = df4.loc[:, [cols4[0], cols4[1], cols4[5], cols4[7]]]
    dfp2 = dfp2.rename(columns={cols4[5]: 'Time to train (h)'})
    dfp2 = dfp2.rename(columns={cols4[7]: 'Throughput (mels/s)'})
    dfp2['Training type'] = 'Mixed precision'

    dff = pd.concat([dfp1, dfp2])

    # st.table(dff)

    cols = list(dff.columns.values)

    fig1 = px.bar(dff,
                  y=cols[3],
                  x=cols[1],
                  color=cols[0],
                  barmode="group",
                  color_discrete_map={'UCloud': ucloud_color,
                                      'IBM PowerAI': pwrai_color,
                                      },
                  hover_data=[cols[1], cols[3]],
                  facet_col=cols[4]
                  )

    fig1.update_xaxes(tickvals=[1, 2, 4], title_text="Number of GPUs", linecolor='black')
    fig1.update_yaxes(linecolor='black', showgrid=True, gridwidth=1, gridcolor='LightGrey')
    fig1.update_layout({'paper_bgcolor': plt_bkg_color, 'plot_bgcolor': plt_bkg_color})

    fig2 = px.bar(dff,
                  y=cols[2],
                  x=cols[1],
                  color=cols[0],
                  barmode="group",
                  color_discrete_map={'UCloud': ucloud_color,
                                      'IBM PowerAI': pwrai_color,
                                      },
                  hover_data=[cols[1], cols[2]],
                  facet_col=cols[4]
                  )

    fig2.update_xaxes(tickvals=[1, 2, 4], title_text="Number of GPUs", linecolor='black')
    fig2.update_yaxes(linecolor='black', showgrid=True, gridwidth=1, gridcolor='LightGrey')
    fig2.update_layout({'paper_bgcolor': plt_bkg_color, 'plot_bgcolor': plt_bkg_color})

    st.plotly_chart(fig1)
    st.plotly_chart(fig2)

elif radio == "Benchmark 5":

    ###############################
    st.subheader("**Benchmark 5**")
    ###############################

    st.markdown("""
    **Category:** 
    Natural Language Processing

    **Model:**
    [Transformer-XL](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/LanguageModeling/Transformer-XL)

    **Framework:**
    PyTorch

    """)

    path_to_file = "training/PyTorch/LanguageModeling/Transformer-XL/results.csv"

    df5 = load_data(path_to_file)
    cols5 = list(df5.columns.values)

    if st.checkbox("Show data benchmark 5"):
        st.table(df5)

    dfp1 = df5.loc[:, [cols5[0], cols5[1], cols5[5], cols5[7]]]
    dfp1 = dfp1.rename(columns={cols5[5]: 'Time to train (min)'})
    dfp1 = dfp1.rename(columns={cols5[7]: 'Throughput (tok/s)'})
    dfp1['Training type'] = 'FP32'

    dfp2 = df5.loc[:, [cols5[0], cols5[1], cols5[6], cols5[8]]]
    dfp2 = dfp2.rename(columns={cols5[6]: 'Time to train (min)'})
    dfp2 = dfp2.rename(columns={cols5[8]: 'Throughput (tok/s)'})
    dfp2['Training type'] = 'Mixed precision'

    dff = pd.concat([dfp1, dfp2])

    # st.table(dff)

    cols = list(dff.columns.values)

    fig1 = px.bar(dff,
                  y=cols[3],
                  x=cols[1],
                  color=cols[0],
                  barmode="group",
                  color_discrete_map={'UCloud': ucloud_color,
                                      'IBM PowerAI': pwrai_color,
                                      'NVIDIA DGX-1': dgx1_color,
                                      'NVIDIA DGX-2': dgx2_color
                                      },
                  hover_data=[cols[1], cols[3]],
                  facet_col=cols[4]
                  )

    fig1.update_xaxes(tickvals=[1, 2, 4, 8, 16], title_text="Number of GPUs", linecolor='black')
    fig1.update_yaxes(linecolor='black', showgrid=True, gridwidth=1, gridcolor='LightGrey')
    fig1.update_layout({'paper_bgcolor': plt_bkg_color, 'plot_bgcolor': plt_bkg_color})

    st.plotly_chart(fig1)

else:
    description
