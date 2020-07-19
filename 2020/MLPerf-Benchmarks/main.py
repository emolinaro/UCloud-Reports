import pandas as pd
import streamlit as st
import plotly.express as px

"""
# UCloud Report

## Deep Learning Benchmark Tests

### Latest update:
July 19, 2020

### Author:
Emiliano Molinaro Ph.D. \n
Computational Scientist \n
eScience Center \n
Syddansk Universitet

### Desciption:

This report summarizes the results of performance tests done on the UCloud interactive HPC platform and the IBM PowerAI 
system, based on [MLPerf training benchmarks](https://mlperf.org/training-overview/#overview). Each benchmark measures 
the wallclock time required to train a model on the specified dataset to achieve the specified quality target. 
The tests are done with NVIDIA CUDA-X software stack running on NVIDIA Volta GPUs.
The results are compared with the performance of NVIDIA DGX-1/DGX-2 systems reported 
[here](https://github.com/NVIDIA/DeepLearningExamples).

"""

st.write("-------")

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

df1 = pd.read_csv(path_to_file)
cols1 = list(df1.columns.values)

if st.checkbox("Show data benchmark 1"):
	st.table(df1)

# fig1 = px.bar(df1,
#               y=cols1[6],
#               x=cols1[1],
#               color=cols1[0],
#               barmode="group",
#               color_discrete_map={'UCloud': 'blue',
#                                   'IBM PowerAI': 'red',
#                                   'NVIDIA DGX-1': 'green',
#                                   'NVIDIA DGX-2': 'orange'},
#               title=cols1[6],
#               hover_data=[cols1[1], cols1[6]],
#               )
# fig1.update_xaxes(tickvals=[1, 2, 3, 4, 8, 16], title_text="Number of GPUs")
# fig1.update_yaxes(title_text=None)
#
# fig2 = px.bar(df1,
#               y=cols1[7],
#               x=cols1[1],
#               color=cols1[0],
#               barmode="group",
#               color_discrete_map={'UCloud': 'blue',
#                                   'IBM PowerAI': 'red',
#                                   'NVIDIA DGX-1': 'green',
#                                   'NVIDIA DGX-2': 'orange'},
#               title=cols1[7],
#               hover_data=[cols1[1], cols1[7]],
#               )
# fig2.update_xaxes(tickvals=[1, 2, 3, 4, 8, 16], title_text="Number of GPUs")
# fig2.update_yaxes(title_text=None)
#
# st.plotly_chart(fig1)
# st.plotly_chart(fig2)

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
              color_discrete_map={'UCloud': 'blue',
                                  'IBM PowerAI': 'red',
                                  'NVIDIA DGX-1': 'green',
                                  'NVIDIA DGX-2': 'orange'},
              hover_data=[cols[1], cols[3]],
              facet_col=cols[4]
              )

fig1.update_xaxes(tickvals=[1, 2, 3, 4, 8, 16], title_text="Number of GPUs")

fig2 = px.bar(dff,
              y=cols[2],
              x=cols[1],
              color=cols[0],
              barmode="group",
              color_discrete_map={'UCloud': 'blue',
                                  'IBM PowerAI': 'red',
                                  'NVIDIA DGX-1': 'green',
                                  'NVIDIA DGX-2': 'orange',
                                  },
              hover_data=[cols[1], cols[2]],
              facet_col=cols[4]
              )

fig2.update_xaxes(tickvals=[1, 2, 3, 4, 8, 16], title_text="Number of GPUs")

st.plotly_chart(fig1)
st.plotly_chart(fig2)

st.write("-------")

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

df2 = pd.read_csv(path_to_file)
cols2 = list(df2.columns.values)

if st.checkbox("Show data benchmark 2"):
	st.table(df2)

# fig1 = px.bar(df2,
#               y=cols2[5],
#               x=cols2[1],
#               color=cols2[0],
#               barmode="group",
#               color_discrete_map={'UCloud': 'blue',
#                                   'IBM PowerAI': 'red',
#                                   'NVIDIA DGX-1': 'green',
#                                   },
#               title=cols2[5],
#               hover_data=[cols2[1], cols2[5]],
#               )
# fig1.update_xaxes(tickvals=[1, 2, 4, 8], title_text="Number of GPUs")
# fig1.update_yaxes(title_text=None)
#
# fig2 = px.bar(df2,
#               y=cols2[6],
#               x=cols2[1],
#               color=cols2[0],
#               barmode="group",
#               color_discrete_map={'UCloud': 'blue',
#                                   'IBM PowerAI': 'red',
#                                   'NVIDIA DGX-1': 'green',
#                                   },
#               title=cols2[6],
#               hover_data=[cols2[1], cols2[6]],
#               )
# fig2.update_xaxes(tickvals=[1, 2, 4, 8], title_text="Number of GPUs")
# fig2.update_yaxes(title_text=None)
#
# st.plotly_chart(fig1)
# st.plotly_chart(fig2)

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
              color_discrete_map={'UCloud': 'blue',
                                  'IBM PowerAI': 'red',
                                  'NVIDIA DGX-1': 'green',
                                  },
              hover_data=[cols[1], cols[3]],
              facet_col=cols[4]
              )

fig1.update_xaxes(tickvals=[1, 2, 4, 8], title_text="Number of GPUs")

fig2 = px.bar(dff,
              y=cols[2],
              x=cols[1],
              color=cols[0],
              barmode="group",
              color_discrete_map={'UCloud': 'blue',
                                  'IBM PowerAI': 'red',
                                  'NVIDIA DGX-1': 'green',
                                  },
              hover_data=[cols[1], cols[2]],
              orientation='v',
              facet_col=cols[4]
              )

fig2.update_xaxes(tickvals=[1, 2, 4, 8], title_text="Number of GPUs")

st.plotly_chart(fig1)
st.plotly_chart(fig2)

st.write("-------")

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

df3 = pd.read_csv(path_to_file)
cols3 = list(df3.columns.values)

if st.checkbox("Show data benchmark 3"):
	st.table(df3)

# fig1 = px.bar(df3,
#               y=cols3[7],
#               x=cols3[1],
#               color=cols3[0],
#               barmode="group",
#               color_discrete_map={'UCloud': 'blue',
#                                   'IBM PowerAI': 'red',
#                                   'NVIDIA DGX-1': 'green',
#                                   'NVIDIA DGX-2': 'orange'
#                                   },
#               title=cols3[7],
#               hover_data=[cols3[1], cols3[7]],
#               )
# fig1.update_xaxes(tickvals=[1, 2, 4, 8, 16], title_text="Number of GPUs")
# fig1.update_yaxes(title_text=None)
#
# fig2 = px.bar(df3,
#               y=cols3[8],
#               x=cols3[1],
#               color=cols3[0],
#               barmode="group",
#               color_discrete_map={'UCloud': 'blue',
#                                   'IBM PowerAI': 'red',
#                                   'NVIDIA DGX-1': 'green',
#                                   'NVIDIA DGX-2': 'orange'
#                                   },
#               title=cols3[8],
#               hover_data=[cols3[1], cols3[8]],
#               )
# fig2.update_xaxes(tickvals=[1, 2, 4, 8, 16], title_text="Number of GPUs")
# fig2.update_yaxes(title_text=None)
#
# st.plotly_chart(fig1)
# st.plotly_chart(fig2)

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
              color_discrete_map={'UCloud': 'blue',
                                  'IBM PowerAI': 'red',
                                  'NVIDIA DGX-1': 'green',
                                  'NVIDIA DGX-2': 'orange'},
              hover_data=[cols[1], cols[3]],
              facet_col=cols[4]
              )

fig1.update_xaxes(tickvals=[1, 2, 4, 8, 16], title_text="Number of GPUs")

fig2 = px.bar(dff,
              y=cols[2],
              x=cols[1],
              color=cols[0],
              barmode="group",
              color_discrete_map={'UCloud': 'blue',
                                  'IBM PowerAI': 'red',
                                  'NVIDIA DGX-1': 'green',
                                  'NVIDIA DGX-2': 'orange',
                                  },
              hover_data=[cols[1], cols[2]],
              facet_col=cols[4]
              )

fig2.update_xaxes(tickvals=[1, 2, 4, 8, 16], title_text="Number of GPUs")

st.plotly_chart(fig1)
st.plotly_chart(fig2)

st.write("-------")

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

df4 = pd.read_csv(path_to_file)
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
              color_discrete_map={'UCloud': 'blue',
                                  'IBM PowerAI': 'red',
                                  },
              hover_data=[cols[1], cols[3]],
              facet_col=cols[4]
              )

fig1.update_xaxes(tickvals=[1, 2, 4], title_text="Number of GPUs")

fig2 = px.bar(dff,
              y=cols[2],
              x=cols[1],
              color=cols[0],
              barmode="group",
              color_discrete_map={'UCloud': 'blue',
                                  'IBM PowerAI': 'red',
                                  },
              hover_data=[cols[1], cols[2]],
              facet_col=cols[4]
              )

fig2.update_xaxes(tickvals=[1, 2, 4], title_text="Number of GPUs")

st.plotly_chart(fig1)
st.plotly_chart(fig2)
