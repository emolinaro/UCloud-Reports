import pandas as pd
import streamlit as st
import plotly.express as px


@st.cache
def load_data(file):
    data = pd.read_csv(file,
                       na_filter=True,
                       na_values=[' -', '-'],
                       keep_default_na=False)
    return data


#####################
### HTML SETTINGS ###
#####################

ucloud_color = '#006AFF'
boston_color = '#febb19'
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


##########################
### Plotting functions ###
##########################

def gen_line_chart(df, x_axis, y_axis):
    """
        Generate line charts.
    """

    cols = list(df.columns.values)

    fig = px.line(df,
                  x=x_axis,
                  y=y_axis,
                  facet_col=cols[1],
                  color=cols[0],
                  color_discrete_map={'UCloud': ucloud_color,
                                      'Boston Server': boston_color,
                                      })

    fig.update_xaxes(tickvals=[8, 16, 32, 64, 128, 256],
                     title_text="batch size",
                     linecolor='black',
                     type='log',
                     showgrid=True,
                     gridwidth=1,
                     gridcolor='LightGrey')

    fig.update_yaxes(linecolor='black',
                     showgrid=True,
                     gridwidth=1,
                     gridcolor='LightGrey')

    fig.update_layout({'paper_bgcolor': plt_bkg_color,
                       'plot_bgcolor': plt_bkg_color})

    fig.update_traces(mode='lines+markers',
                      marker_symbol='hexagram',
                      marker_size=9)

    return fig


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
November 13, 2020

### Report:
2020-3

### Author:
Emiliano Molinaro, Ph.D. (<molinaro@imada.sdu.dk>)\n
Computational Scientist \n
Research Support Lead \n
SDU eScience

### Description:
The purpose of this study is to test the 
performance of the DGX A100 server provided by [Boston Limited](https://www.boston.co.uk/default.aspx) for the distributed training of
deep learning models. The results are compared with equivalent simulations performed on the UCloud system.

In all the tests we train the same model for 3 epochs and we fix the data batch size on a single device. 
We use three different datasets. 
The reference literature for the model and the datasets is reported [here](https://openreview.net/pdf?id=rJ4km2R5t7).

The training process is executed in the following modes:
- single-precision floating arithmetic (FP32) on NVIDIA V100 GPUs
- native TensorFloat-32 (TF32) precison on NVIDIA A100 GPUs 
- automatic mixed precision (AMP) on all devices

More information can be found in this blog [post](https://blogs.nvidia.com/blog/2020/05/14/tensorfloat-32-precision-format/).

### Specs:

#### 1. UCloud
`u1-gpu-4` machine type: 
- 4x NVIDIA Tesla V100 GPUs SXM2 32GB 
- 64 CPU cores
- 180 GB of memory

#### 2. Boston Server
NVIDIA DGX A100:
- 8x NVIDIA Tesla A100 GPUs SXM4 40GB
- 6x NVIDIA NVSWITCHES
- 9x Mellanox ConnectX-6 200Gb/S Network Interface
- Dual 64-core AMD CPUs and 1TB System Memory
- 15TB Gen4 NVMe SSD

"""

#################
### SIDE MENU ###
#################

# logo = Image.open('figs/logo_esc.png')
# st.sidebar.image(logo, format='PNG', width=50)
st.sidebar.title("Benchmark Models")

radio = st.sidebar.radio(label="", options=["Description",
                                            "Dataset 1",
                                            "Dataset 2",
                                            "Dataset 3",
                                            ])

if radio == "Dataset 1":

    ###############################
    st.subheader("**Dataset 1**")
    ###############################

    st.markdown("""
        **Category:** 
        Text Classification 

        **Model:**
        [GLUE](https://github.com/huggingface/transformers/tree/master/examples/text-classification)
        
        **Dataset:**
        The Microsoft Research Paraphrase Corpus (MRPC), with 3.7k sentence pairs
        
        **Framework:**
        PyTorch

        """)

    path_to_file = "training/PyTorch/TextClassification/MRPC/results.csv"

    df1 = load_data(path_to_file)
    cols1 = list(df1.columns.values)

    if st.checkbox("Show dataset 1"):
        st.table(df1)

    dfp1 = df1.loc[:, [cols1[0], cols1[1], cols1[2], cols1[3], cols1[4]]]
    dfp1 = dfp1.rename(columns={cols1[4]: 'Time to train (s)'})
    dfp1 = dfp1.rename(columns={cols1[3]: 'Accuracy'})
    dfp1['Training type'] = 'FP32/TF32'

    dfp2 = df1.loc[:, [cols1[0], cols1[1], cols1[2], cols1[5], cols1[6]]]
    dfp2 = dfp2.rename(columns={cols1[6]: 'Time to train (s)'})
    dfp2 = dfp2.rename(columns={cols1[5]: 'Accuracy'})
    dfp2['Training type'] = 'AMP'

    dff = pd.concat([dfp1, dfp2])
    cols = list(dff.columns.values)

    train_type = st.selectbox("Select training type",
                              ('FP32/TF32', 'AMP'))

    dff_t = dff[dff['Training type'] == train_type].drop(columns='Training type')
    cols_t = list(dff_t.columns.values)

    # st.table(dff_t)

    ## Create line charts
    fig1 = gen_line_chart(dff_t, cols_t[2], cols_t[4])
    fig2 = gen_line_chart(dff_t, cols_t[2], cols_t[3])

    st.plotly_chart(fig1)
    st.plotly_chart(fig2)

elif radio == "Dataset 2":

    ###############################
    st.subheader("**Dataset 2**")
    ###############################

    st.markdown("""
            **Category:** 
            Text Classification 

            **Model:**
            [GLUE](https://github.com/huggingface/transformers/tree/master/examples/text-classification)

            **Dataset:**
            The Quora Question Pairs (QQP) collection, with 364k sentence pairs

            **Framework:**
            PyTorch

            """)

    path_to_file = "training/PyTorch/TextClassification/QQP/results.csv"

    df2 = load_data(path_to_file)
    cols2 = list(df2.columns.values)

    if st.checkbox("Show dataset 2"):
        st.table(df2)

    dfp1 = df2.loc[:, [cols2[0], cols2[1], cols2[2], cols2[3], cols2[4]]]
    dfp1 = dfp1.rename(columns={cols2[4]: 'Time to train (s)'})
    dfp1 = dfp1.rename(columns={cols2[3]: 'Accuracy'})
    dfp1['Training type'] = 'FP32/TF32'

    dfp2 = df2.loc[:, [cols2[0], cols2[1], cols2[2], cols2[5], cols2[6]]]
    dfp2 = dfp2.rename(columns={cols2[6]: 'Time to train (s)'})
    dfp2 = dfp2.rename(columns={cols2[5]: 'Accuracy'})
    dfp2['Training type'] = 'AMP'

    dff = pd.concat([dfp1, dfp2])
    cols = list(dff.columns.values)

    train_type = st.selectbox("Select training type",
                              ('FP32/TF32', 'AMP'))

    dff_t = dff[dff['Training type'] == train_type].drop(columns='Training type')
    cols_t = list(dff_t.columns.values)

    ## Create line charts
    fig1 = gen_line_chart(dff_t, cols_t[2], cols_t[4])
    fig2 = gen_line_chart(dff_t, cols_t[2], cols_t[3])
    fig2.update_yaxes(range=[0.86, 0.9])

    st.plotly_chart(fig1)
    st.plotly_chart(fig2)

elif radio == "Dataset 3":

    ###############################
    st.subheader("**Dataset 3**")
    ###############################

    st.markdown("""
                **Category:** 
                Text Classification 

                **Model:**
                [GLUE](https://github.com/huggingface/transformers/tree/master/examples/text-classification)

                **Dataset:**
                The Multi-Genre Natural Language Inference (MNLI) corpus, with 393k sentence pairs

                **Framework:**
                PyTorch

                """)

    path_to_file = "training/PyTorch/TextClassification/MNLI/results.csv"

    df3 = load_data(path_to_file)
    cols3 = list(df3.columns.values)

    if st.checkbox("Show dataset 3"):
        st.table(df3)

    dfp1 = df3.loc[:, [cols3[0], cols3[1], cols3[2], cols3[3], cols3[4]]]
    dfp1 = dfp1.rename(columns={cols3[4]: 'Time to train (s)'})
    dfp1 = dfp1.rename(columns={cols3[3]: 'Accuracy'})
    dfp1['Training type'] = 'FP32/TF32'

    dfp2 = df3.loc[:, [cols3[0], cols3[1], cols3[2], cols3[5], cols3[6]]]
    dfp2 = dfp2.rename(columns={cols3[6]: 'Time to train (s)'})
    dfp2 = dfp2.rename(columns={cols3[5]: 'Accuracy'})
    dfp2['Training type'] = 'AMP'

    dff = pd.concat([dfp1, dfp2])
    cols = list(dff.columns.values)

    train_type = st.selectbox("Select training type",
                              ('FP32/TF32', 'AMP'))

    dff_t = dff[dff['Training type'] == train_type].drop(columns='Training type')
    cols_t = list(dff_t.columns.values)

    ## Create line charts
    fig1 = gen_line_chart(dff_t, cols_t[2], cols_t[4])
    fig2 = gen_line_chart(dff_t, cols_t[2], cols_t[3])
    fig2.update_yaxes(range=[0.8, 0.85])

    st.plotly_chart(fig1)
    st.plotly_chart(fig2)

else:
    description
