FROM quay.io/fenicsproject/stable:latest

RUN apt update && apt -y upgrade
RUN apt install -y python3-h5py

RUN pip install matplotlib cmcrameri numpy tqdm dataclasses-json scikit-learn