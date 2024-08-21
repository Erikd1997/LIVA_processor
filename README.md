# LIVA_processor
Repository for the Local Interrogation Volume Approach processor, for the computation of skin friction topology (and to a certain extent magnitude) from Lagrangian Particle Tracking data and object registration.

## Setting up the required Python packages
The Python packages can be installed into a `conda` virtual environment using the `requirements_conda.yml` file in the root directory. You can create a new virtual environment via
```
conda env create --name <env> --file requirements_conda.yml
```

If you prefer to use `pip` virtual environemnts (venv), you can use the `requirements_pip.txt` file instead. Then create a new `venv` with the necessary dependencies with
```
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```
> [!NOTE]
> Do not forget to change `<env>`/`env` in the above examples to an appropriate name, for example `LIVA_processor`

## Getting started
1. Before processing
	* Obtaining the STL file
	* Creating the outputmesh with `Gmsh`, version 4.13.1 for Windows available in `./tools`
	* Obtaining the fluid data, bin-based or tracer-based.
2. Starting up the application

## Different Modules

## Examples
