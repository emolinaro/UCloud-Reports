# Dockerfile

ARG BASE_CONTAINER=continuumio/miniconda3

FROM $BASE_CONTAINER

MAINTAINER Emiliano Molinaro <molinaro@imada.sdu.dk>  

LABEL software="Streamlit" \
	  author="Adrien Treuille, Thiago Teixeira, Amanda Kelly" \
	  version="0.63.1" \
	  license="Apache License 2.0" \
	  description="Streamlit is an open-source Python library that makes it easy to build beautiful custom web-apps for machine learning and data science."

ENV REFRESHED_AT 2020-07-19

ARG user=${user:-"ucloud"}

## Create default user
### Debian
RUN adduser --shell /bin/bash --uid 11042 --disabled-password --gecos "" $user

## Create working directory
RUN mkdir /work && \
	chown -R $user:$user /work /opt/conda

## Install software
###################

RUN apt-get update && \
	apt-get install -y \
	build-essential \
	apt-utils \
	git && \
	rm -rf /var/lib/apt/lists/* 

COPY --chown=$user start_webapp /usr/bin/

USER 11042

COPY --chown=$user environment.yml /tmp/
RUN  conda update -n base -c defaults conda
RUN  conda config --add channels conda-forge
RUN	 conda config --add channels bioconda 
RUN  conda env update -f /tmp/environment.yml && conda clean -a

RUN mkdir -p /home/$user/.streamlit && \
	bash -c 'echo -e " \
	[general]\n \
	email = \"\"\n \
	" > /home/$user/.streamlit/credentials.toml' && \
	bash -c 'echo -e " \
	[server]\n \
	enableCORS = false\n \
	enableXsrfProtection = false\n \
	" > /home/$user/.streamlit/config.toml'


## Set shell prompt
RUN echo 'PS1="\[\033[01;31m\]\u\[\033[01;37m\]:\[\033[01;34m\]\w\[\033[00m\]\$ "' >> /home/$user/.bashrc

WORKDIR /work

EXPOSE 8051

# ENTRYPOINT start_webapp
# CMD





