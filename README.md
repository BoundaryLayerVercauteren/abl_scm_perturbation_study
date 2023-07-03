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
docker run -ti -v "$PWD:/home/fenics/shared" abl_scm_venv
cd shared
```

3. Run the model
```bash
python3 main.py
```