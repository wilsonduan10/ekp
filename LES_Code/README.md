## Overview
This folder includes all the calibration files that pertain to the use of EKI with LES data, which can be found here: https://data.caltech.edu/records/a59sz-z5n11. The data is split into different cfSites and months, marking the location of the data collected and the month it was collected at, respectively. The LES data consists of data mostly belonging to the unstable surface layer, hence we calibrate only b_m and b_h of the Businger function parameters.

![](../assets/LESdata.png)

## How to run
The files must be run as follows:
- First, make sure you are in the root of the repository in the terminal.
- Second, run the command: julia --project
- Third, instantiate all dependencies.
- Lastly, type the line: include("LES_Code/<`filename`>")

## Contents
The core files of this folder are `load_data.jl` and `physical_model.jl` – `load_data.jl` loads in the cfSite data, and `physical_model.jl` represents the surface flux model. Aside from that, there exist files that use different observables to perform calibration, as well as files for data analysis and sensitivity analysis. 

### `load_data.jl`
This file defines a dataset struct that holds all of the data relevant for calibration, such as wind speed, specific humidity, and temperature at different timesteps and different heights. It also defines a function `create_dataframe`, which, when given a cfSite and month to retrieve data from, it returns a dataframe populated with the data given from the file with the given cfSite and month. It contains an optional boolean argument extrapolate_surface, which allows the user to extrapolate the values of air density and total specific humidity at the surface in the case that it is not given by the dataset. Since the LES data does not provide these values, the extrapolate_surface parameter defaults to true.

### `physical_model.jl`
- PhaseEquil options
- ValuesOnly, Fluxes, etc options (defaulted to ValuesOnly)
- UF option (defaulted to Businger)
- parameterTypes option (defaulted to b_m, b_h)
- purpose: given data and SF parameters, return the surface conditions at each timestep + height
    - these surface conditions are to be fed into some H map to result in observables

### `businger_calibration.jl`
- explain what the file does when it is run
    - calibration + images output to folder
- 

### `perfect_model.jl`
