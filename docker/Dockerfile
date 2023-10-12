FROM quay.io/fenicsproject/stable:latest

RUN apt update && apt -y upgrade && apt install --fix-missing
RUN apt install -y python3-h5py
RUN pip install --upgrade pip
RUN DEBIAN_FRONTEND=noninteractive TZ=8/36 apt install -y texlive texlive-latex-extra texlive-fonts-recommended dvipng
RUN DEBIAN_FRONTEND=noninteractive TZ=8/36 apt install -y cm-super

RUN pip install matplotlib cmcrameri numpy tqdm dataclasses-json scikit-learn SciencePlots pandas pytest