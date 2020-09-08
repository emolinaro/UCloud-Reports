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
                                      'IBM PowerAI': pwrai_color,
                                      })

    fig.update_xaxes(tickvals=[8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4092],
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


def gen_bar_chart(df, x_axis, y_axis, options):
    """
        Generate bar charts
    """

    fig = px.bar(df.loc[options[0]],
                 x=x_axis,
                 y=y_axis,
                 barmode="group",
                 orientation='h',
                 color=cols_t[0],
                 color_discrete_map={'UCloud': ucloud_color,
                                     'IBM PowerAI': pwrai_color,
                                     })

    for option in options[1:]:
        fig.add_trace(px.bar(dff.loc[option],
                             x=x_axis,
                             y=y_axis,
                             barmode="group",
                             orientation='h',
                             color=cols_t[0],
                             color_discrete_map={'UCloud': ucloud_color,
                                                 'IBM PowerAI': pwrai_color,
                                                 }).data[0])

    fig.update_yaxes(tickvals=[1, 2, 4],
                     title_text="Number of GPUs",
                     linecolor='black')

    fig.update_xaxes(linecolor='black',
                     showgrid=True,
                     gridwidth=1,
                     gridcolor='LightGrey')

    fig.update_layout({'paper_bgcolor': plt_bkg_color,
                       'plot_bgcolor': plt_bkg_color})

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
September 8, 2020

### Report:
2020-2

### Author:
Emiliano Molinaro (<molinaro@imada.sdu.dk>)\n
Computational Scientist \n
SDU eScience

### Description:
The purpose of this set of simulations is to test the 
[IBM Large Model Support](https://www.ibm.com/support/knowledgecenter/SS5SF7_1.7.0/navigation/wmlce_getstarted_pytorch.html#wmlce_getstarted_pytorch__lms_section) 
(LMS) Python library. 
This feature allows to move layers of a model between GPU and CPU to overcome memory limits during 
the training phase. The latter are tipically due to:
- Model depth/complexity
- Data size, e.g. high-resolution images
- Large batch size

In both the benchmarks reported here the model is trained for 3 epochs and different values of the data batch size.
The training process is executed in 
- single-precision floating arithmetic (FP32)
- automatic mixed precision (APEX)
- LMS (only for the IBM PowerAI system)

### Specs:

#### 1. UCloud
`u1-gpu-4` machine type: 
- 4 NVIDIA Volta GPUs 
- 78 CPU cores
- 185 GB of memory

#### 2. PowerAI
POWER9 Server AC922 w/ 4x V100 GPUs and NVLink to GPUs.


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
                                            ])

if radio == "Benchmark 1":

    ###############################
    st.subheader("**Benchmark 1**")
    ###############################

    st.markdown("""
        **Category:** 
        Text Classification 

        **Model:**
        [GLUE](https://github.com/huggingface/transformers/tree/master/examples/text-classification)
        
        **Dataset:**
        MRPC
        
        **Framework:**
        PyTorch

        """)

    path_to_file = "training/PyTorch/TextClassification/MRPC/results.csv"

    df1 = load_data(path_to_file)
    cols1 = list(df1.columns.values)

    if st.checkbox("Show data benchmark 1"):
        st.table(df1)

    dfp1 = df1.loc[:, [cols1[0], cols1[1], cols1[2], cols1[3], cols1[4]]]
    dfp1 = dfp1.rename(columns={cols1[4]: 'Time to train (s)'})
    dfp1 = dfp1.rename(columns={cols1[3]: 'Accuracy'})
    dfp1['Training type'] = 'FP32'

    dfp2 = df1.loc[:, [cols1[0], cols1[1], cols1[2], cols1[5], cols1[6]]]
    dfp2 = dfp2.rename(columns={cols1[6]: 'Time to train (s)'})
    dfp2 = dfp2.rename(columns={cols1[5]: 'Accuracy'})
    dfp2['Training type'] = 'APEX'

    dfp3 = df1.loc[:, [cols1[0], cols1[1], cols1[2], cols1[7], cols1[8]]]
    dfp3 = dfp3.rename(columns={cols1[8]: 'Time to train (s)'})
    dfp3 = dfp3.rename(columns={cols1[7]: 'Accuracy'})
    dfp3['Training type'] = 'LMS'

    dff = pd.concat([dfp1, dfp2, dfp3])
    cols = list(dff.columns.values)

    train_type = st.selectbox("Select training type",
                              ('FP32', 'APEX', 'LMS'))

    dff_t = dff[dff['Training type'] == train_type].drop(columns='Training type')
    cols_t = list(dff_t.columns.values)

    # st.table(dff_t)

    ## Create line charts
    fig1 = gen_line_chart(dff_t, cols_t[2], cols_t[4])
    fig2 = gen_line_chart(dff_t, cols_t[2], cols_t[3])

    st.plotly_chart(fig1)
    st.plotly_chart(fig2)

    ## Create bar charts

    ### GLOBAL BATCH SIZE = 1024
    option_1 = (dff[cols[0]] == 'IBM PowerAI') & (dff[cols[1]] == 1) & (dff[cols[2]] == 1024) & (dff[cols[5]] == 'LMS')
    option_2 = (dff[cols[0]] == 'UCloud') & (dff[cols[1]] == 4) & (dff[cols[2]] == 256) & (dff[cols[5]] == 'APEX')
    options = [option_1, option_2]

    fig3a = gen_bar_chart(dff, cols[4], cols[1], options)
    fig3a.update_layout({'title_text': 'GLOBAL BATCH SIZE: 1024', 'title_font_color': 'darkgreen', 'title_x': 0.5})
    fig3b = gen_bar_chart(dff, cols[3], cols[1], options)

    st.plotly_chart(fig3a)
    st.plotly_chart(fig3b)

    ### GLOBAL BATCH SIZE = 512
    option_1 = (dff[cols[0]] == 'IBM PowerAI') & (dff[cols[1]] == 1) & (dff[cols[2]] == 512) & (dff[cols[5]] == 'LMS')
    option_2 = (dff[cols[0]] == 'UCloud') & (dff[cols[1]] == 2) & (dff[cols[2]] == 256) & (dff[cols[5]] == 'APEX')
    option_3 = (dff[cols[0]] == 'UCloud') & (dff[cols[1]] == 4) & (dff[cols[2]] == 128) & (dff[cols[5]] == 'APEX')

    options = [option_1, option_2, option_3]

    fig3c = gen_bar_chart(dff, cols[4], cols[1], options)
    fig3c.update_layout({'title_text': 'GLOBAL BATCH SIZE: 512', 'title_font_color': 'darkgreen', 'title_x': 0.5})
    fig3d = gen_bar_chart(dff, cols[3], cols[1], options)

    st.plotly_chart(fig3c)
    st.plotly_chart(fig3d)

    # from plotly.subplots import make_subplots
    #
    # fig = make_subplots(rows=1, cols=2, shared_yaxes=True)
    #
    # for k in range(len(options)):
    #     fig.add_trace(fig3a['data'][k], row=1, col=1)
    #     fig.add_trace(fig3b['data'][k], row=1, col=2)
    #
    # fig.update_yaxes(tickvals=[1, 2, 4],
    #                  title_text="Number of GPUs",
    #                  linecolor='black')
    #
    # fig.update_xaxes(linecolor='black',
    #                  showgrid=True,
    #                  gridwidth=1,
    #                  gridcolor='LightGrey')
    #
    # fig.update_layout({'paper_bgcolor': plt_bkg_color,
    #                    'plot_bgcolor': plt_bkg_color},
    #                   showlegend=False)
    #
    # st.plotly_chart(fig)

elif radio == "Benchmark 2":

    ###############################
    st.subheader("**Benchmark 2**")
    ###############################

    st.markdown("""
            **Category:** 
            Text Classification 

            **Model:**
            [GLUE](https://github.com/huggingface/transformers/tree/master/examples/text-classification)

            **Dataset:**
            QQP

            **Framework:**
            PyTorch

            """)

    path_to_file = "training/PyTorch/TextClassification/QQP/results.csv"

    df2 = load_data(path_to_file)
    cols2 = list(df2.columns.values)

    if st.checkbox("Show data benchmark 2"):
        st.table(df2)

    dfp1 = df2.loc[:, [cols2[0], cols2[1], cols2[2], cols2[3], cols2[4]]]
    dfp1 = dfp1.rename(columns={cols2[4]: 'Time to train (s)'})
    dfp1 = dfp1.rename(columns={cols2[3]: 'Accuracy'})
    dfp1['Training type'] = 'FP32'

    dfp2 = df2.loc[:, [cols2[0], cols2[1], cols2[2], cols2[5], cols2[6]]]
    dfp2 = dfp2.rename(columns={cols2[6]: 'Time to train (s)'})
    dfp2 = dfp2.rename(columns={cols2[5]: 'Accuracy'})
    dfp2['Training type'] = 'APEX'

    dfp3 = df2.loc[:, [cols2[0], cols2[1], cols2[2], cols2[7], cols2[8]]]
    dfp3 = dfp3.rename(columns={cols2[8]: 'Time to train (s)'})
    dfp3 = dfp3.rename(columns={cols2[7]: 'Accuracy'})
    dfp3['Training type'] = 'LMS'

    dff = pd.concat([dfp1, dfp2, dfp3])
    cols = list(dff.columns.values)

    train_type = st.selectbox("Select training type",
                              ('FP32', 'APEX', 'LMS'))

    dff_t = dff[dff['Training type'] == train_type].drop(columns='Training type')
    cols_t = list(dff_t.columns.values)

    ## Create line charts
    fig1 = gen_line_chart(dff_t, cols_t[2], cols_t[4])
    fig2 = gen_line_chart(dff_t, cols_t[2], cols_t[3])

    st.plotly_chart(fig1)
    st.plotly_chart(fig2)

    ## Create bar charts

    ### GLOBAL BATCH SIZE = 1024
    option_1 = (dff[cols[0]] == 'IBM PowerAI') & (dff[cols[1]] == 1) & (dff[cols[2]] == 1024) & (dff[cols[5]] == 'LMS')
    option_2 = (dff[cols[0]] == 'UCloud') & (dff[cols[1]] == 4) & (dff[cols[2]] == 256) & (dff[cols[5]] == 'APEX')
    options = [option_1, option_2]

    fig3a = gen_bar_chart(dff, cols[4], cols[1], options)
    fig3a.update_layout({'title_text': 'GLOBAL BATCH SIZE: 1024', 'title_font_color': 'darkgreen', 'title_x': 0.5})
    fig3b = gen_bar_chart(dff, cols[3], cols[1], options)

    st.plotly_chart(fig3a)
    st.plotly_chart(fig3b)

    ### GLOBAL BATCH SIZE = 512
    option_1 = (dff[cols[0]] == 'IBM PowerAI') & (dff[cols[1]] == 1) & (dff[cols[2]] == 512) & (dff[cols[5]] == 'LMS')
    option_2 = (dff[cols[0]] == 'UCloud') & (dff[cols[1]] == 2) & (dff[cols[2]] == 256) & (dff[cols[5]] == 'APEX')
    option_3 = (dff[cols[0]] == 'UCloud') & (dff[cols[1]] == 4) & (dff[cols[2]] == 128) & (dff[cols[5]] == 'APEX')

    options = [option_1, option_2, option_3]

    fig3c = gen_bar_chart(dff, cols[4], cols[1], options)
    fig3c.update_layout({'title_text': 'GLOBAL BATCH SIZE: 512', 'title_font_color': 'darkgreen', 'title_x': 0.5})
    fig3d = gen_bar_chart(dff, cols[3], cols[1], options)

    st.plotly_chart(fig3c)
    st.plotly_chart(fig3d)

else:
    description
