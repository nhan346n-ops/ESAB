FROM ubuntu:20.04
LABEL maintainer="<globe-assistance@ifremer.fr>"

# Generate locale C.UTF-8 for postgres and general locale data
ENV LANG C.UTF-8

RUN apt-get update && apt-get install -y locales && rm -rf /var/lib/apt/lists/* \
    && localedef -i en_US -c -f UTF-8 -A /usr/share/locale/locale.alias en_US.UTF-8
ENV LANG en_US.utf8



# CI_USER user should have a > 1000 gid to ease uid/gid mapping in docker
RUN addgroup --gid 1001 ci_user \
    && adduser --system --home /home/ci_user --gid 1001 --uid 1001 --quiet  ci_user

RUN mkdir -p /home/ci_user

RUN chown -R ci_user /home/ci_user

# Install basic dev tools
RUN set -x; \
    apt-get update \
    && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    g++ \
    git \
    && apt-get -y install -f --no-install-recommends \
    && apt-get upgrade -y

USER ci_user

# Install conda
RUN set -x; \
    cd /tmp && \
    curl -o miniconda3.sh -sSL  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh &&\
    chmod +x miniconda3.sh && \
    ./miniconda3.sh -b -p ~/miniconda

RUN set -x; \
    . ~/miniconda/bin/activate && \
    conda init bash && \
    conda install -n base conda-pack -c conda-forge -y

# Git workaround because of self-signed certificates
RUN git config --global http.sslVerify false

# Cleaning
USER root

RUN  apt-get -yq clean && \
    apt-get -yq autoremove && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf /tmp/*

USER ci_user
