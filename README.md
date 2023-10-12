# Atmospheric Boundary Layer Single-Column Model

The code in this repository is used to study the sensitivity of the nocturnal and polar atmospheric boundary layer to transient phenomena. 


## How to set up and run the model
1. Clone repository
```bash
git clone git@github.com:am-kaiser/ABL_SCM_1.5_order.git
cd ABL_SCM_1.5_order
```
2. Create and activate environment (using docker)
```bash
cd docker
docker build . -t abl_scm_venv
cd ..
docker run --rm -ti -v "$PWD:/home/fenics/shared" abl_scm_venv
cd shared
```
Note: To create environment on an external machine which doesn't have docker but singularity create tar file from docker 
image on local machine with
```bash
docker save abl_scm_venv > docker/abl_scm_venv.tar
```
and then move this tar file with scp. On the external machine create singularity environment from tar file with
```bash
cd docker
singularity build abl_scm_venv.sif docker-archive://abl_scm_venv.tar
```

3. Run the model
```bash
python3 main.py # in docker
singularity exec docker/abl_scm_venv.sif python3 -u main.py # with singularity
```