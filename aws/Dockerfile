FROM amazonlinux:latest

ENV PATH /opt/conda/bin:$PATH
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

RUN yum install -y wget git unzip
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda
RUN ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

RUN conda config --set always_yes yes --set changeps1 no
RUN conda update -q conda
RUN conda config --add channels conda-forge
RUN conda config --add channels anaconda
RUN conda install -y cffi cloudpickle lz4 numba numpy psutil python-xxhash pyyaml requests root scipy six tbb tqdm xxhash boost matplotlib
RUN conda install -c conda-forge awscli make
RUN pip install tensorflow keras awkward uproot dask distributed boto3

ADD fetch_and_run.sh /usr/local/bin/fetch_and_run.sh
RUN yum install -y which tar

WORKDIR /tmp
USER nobody
ENTRYPOINT ["/usr/local/bin/fetch_and_run.sh"]
