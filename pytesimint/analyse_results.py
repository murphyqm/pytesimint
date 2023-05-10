""" Result Analysis Script

This module allows the user to filter and analyse results produced by the
PytesiMINT model, taking 3D heatmaps and filtering them according to
experimentally obtained estimates (see details, Murphy Quinlan et al., 2023),
before saving output as pickles. These can then be loaded and compiled into
a human-readable dataframe.

This tool accepts .npy binary files and parameter files saved as .csv as
inputs.

The module contains the following functions:

    - filtering
    - temp_save_param
    - load_param_pickles
    - param_dict_to_df
    - timeseries
    - filtering_checking_bounds
    - timeseries
    - filtering_expand_bounds
    - param_dict_to_df_checking_bounds


"""

import numpy as np
import pandas as pd
import numpy.ma as ma
import pickle

from . import define_matrix as dm



def filtering(job_no, params, results_folder, iterations=[3, 48, 96, 120]):
    example = params.iloc[job_no]
    run_no = example['id']
    timestep = example['dt']
    Nx = example['Nx']
    Ny = example['Ny']
    Nz = example['Nz']

    Lx = example['Lx']
    Ly = example['Ly']
    Lz = example['Lz']

    d_x = Lx/(Nx -1)

    x_mid, y_mid, z_mid = example['x_mid'], example['y_mid'], example['z_mid']
    r_x, r_y, r_z = example['r_x'], example['r_y'], example['r_z']
    expected_vol = (4.0/3.0)*3.14 *r_x * r_y * r_z
    int_nomask = 0
    ext_mask = 1

    x, y, z, blank_vol = dm.define_grid_3d(Nx, Ny, Nz, Lx, Ly, Lz)

    mask = dm.set_grid_values_3d_rounded(x, y, z, blank_vol,
                                     x_mid, y_mid, z_mid,
                                     r_x, r_y, r_z,
                                     int_nomask, ext_mask)

    no_3mnths = iterations[0]
    no_4years = iterations[1]
    no_8years = iterations[2]
    no_10years = iterations[3]

    # first two
    # after 3 months and 10 years
    # rounding of olivine grains - want approx 50% rounded
    # at 3 months > 1623.15 OR at 10 years > 1573.15

    filename = f"{results_folder}/{run_no}/{run_no}_{no_3mnths}.npy"
    results = np.load(filename)
    masked_result_a = ma.masked_array(results, mask=mask)
    total_size = (masked_result_a>100).sum()
    actual_volume = total_size * d_x * d_x * d_x
    mean_it0 = masked_result_a.mean()

    filename = f"{results_folder}/{run_no}/{run_no}_{no_10years}.npy"
    results = np.load(filename)
    masked_result_b = ma.masked_array(results, mask=mask)
    mean_it3 = masked_result_b.mean()

    rounding = np.bitwise_or(masked_result_a > 1623.15, masked_result_b > 1573.15).sum()
    percent_rounded = 100*rounding/total_size
    print(percent_rounded)

    # Second two
    # after 4 years and 8 years
    # preservation of chemical zoning
    # at 4 years < 1573.15 AND at 8 years < 1373.15
    # Want at least 20 pc

    filename = f"{results_folder}/{run_no}/{run_no}_{no_4years}.npy"
    results = np.load(filename)
    masked_result_a = ma.masked_array(results, mask=mask)
    total_size = (masked_result_a>100).sum()
    mean_it1 = masked_result_a.mean()

    filename = f"{results_folder}/{run_no}/{run_no}_{no_8years}.npy"
    results = np.load(filename)
    masked_result_b = ma.masked_array(results, mask=mask)
    mean_it2 = masked_result_b.mean()

    geochem = np.bitwise_and(masked_result_a < 1573.15, masked_result_b < 1373.15).sum()
    percent_geochem_preserved = 100*geochem/total_size
    print(f"Percent geochem preserved: {percent_geochem_preserved}")
    print(f"Percent rounded: {percent_rounded}")
    print(f"r_x: {r_x}; r_y: {r_y}; r_z = {r_z}")
    print(f"Expected volume: {expected_vol}; actual volume: {actual_volume}; difference: {actual_volume - expected_vol}")
    return run_no, percent_rounded, percent_geochem_preserved, r_x, r_y, expected_vol, actual_volume, mean_it0, mean_it1, mean_it2, mean_it3


def temp_save_param(output, folder):
    filepath = f"{folder}/{output[0]}_filtered.pickle"
    print(f"Saving params to {filepath}")
    with open(filepath, 'wb') as f:
        pickle.dump(output, f)


def load_param_pickles(params, savefolder, startnum, endnum):
    param_dict = {}
    for i in range(startnum, endnum):
        example = params.iloc[i]
        filepath = f"{savefolder}/{example['id']}_filtered.pickle"
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        param_dict[f"{i}"] = list(data)
        print(i)
    return param_dict


def param_dict_to_df(params, param_dict):
    # define column names
    column_names = ["model_id", "percent_rounded",
                    "percent_geochem_preserved",
                    "r_x", "r_y",
                    "expected_vol", "actual_volume",
                    "mean_it0", "mean_it1", "mean_it2", "mean_it3"]
    # turn the dict into a dataframe:
    df = pd.DataFrame.from_dict(param_dict, orient="index", columns=column_names)
    # copy the old param df so we can modify it
    new_params = params.copy(deep=True)
    # delete the  example row 1
    new_params.drop(index=new_params.index[0], axis=0, inplace=True)
    # lcean up the index so it matches our new dataframe
    new_params.reset_index(inplace=True)
    # this adds the new dataframe alongside the old, keeping values, making new cols
    new_params[list(df.columns)] = df.to_numpy()
    # some params may be repeated, delete these
    new_params.T.drop_duplicates().T
    # return the new appended dataframe, to be plotted/saved etc
    return new_params


def timeseries(job_no, params, results_folder,):
    example = params.iloc[job_no]
    run_no = example['id']
    timestep = example['dt']
    iterations = example["iterations"]
    save_iter = example["save_iter"]
    T_S = example["T_S"]
    T_L = example["T_L"]
    
    it_list = np.arange(0, iterations, save_iter)
    seconds = list(it_list * timestep)
    
    Nx = example['Nx']
    Ny = example['Ny']
    Nz = example['Nz']

    Lx = example['Lx']
    Ly = example['Ly']
    Lz = example['Lz']

    d_x = Lx/(Nx - 1)

    x_mid, y_mid, z_mid = example['x_mid'], example['y_mid'], example['z_mid']
    r_x, r_y, r_z = example['r_x'], example['r_y'], example['r_z']
    expected_vol = (4.0/3.0)*3.14 * r_x * r_y * r_z
    int_nomask = 0
    ext_mask = 1

    x, y, z, blank_vol = dm.define_grid_3d(Nx, Ny, Nz, Lx, Ly, Lz)

    mask = dm.set_grid_values_3d_rounded(x, y, z, blank_vol,
                                         x_mid, y_mid, z_mid,
                                         r_x, r_y, r_z,
                                         int_nomask, ext_mask)

    mean_temps = []
    solid_fraction = []
    liquid_fraction = []
    for i in it_list:
        filename = f"{results_folder}/{run_no}/{run_no}_{i}.npy"
        results = np.load(filename)
        masked_result = ma.masked_array(results, mask=mask)
        total_size = (masked_result>100).sum()
        mean_temp = masked_result.mean()
        mean_temps.append(mean_temp)
        frac_solid = (masked_result<T_S).sum()
        frac_liquid = (masked_result>T_L).sum()
        percent_solid = 100*frac_solid/total_size
        percent_liquid = 100*frac_liquid/total_size
        solid_fraction.append(percent_solid)
        liquid_fraction.append(percent_liquid)
    
    results_dict = {}
    results_dict["Mean temps"] = list(mean_temps)
    results_dict["Solid frac"] = list(solid_fraction)
    results_dict["Liquid fraction"] = list(liquid_fraction)
    
    return results_dict, run_no


def filtering_checking_bounds(job_no, params, results_folder, iterations=[3, 48, 96, 120]):
    #""" Translate bounds by a certain amount up or down"""
    example = params.iloc[job_no]
    run_no = example['id']
    timestep = example['dt']
    Nx = example['Nx']
    Ny = example['Ny']
    Nz = example['Nz']

    Lx = example['Lx']
    Ly = example['Ly']
    Lz = example['Lz']

    d_x = Lx/(Nx - 1)

    temps = np.array([1623.15, 1573.15, 1373.15])
    temps_p10pc = temps + (temps * 0.01)
    temps_m10pc = temps - (temps * 0.01)
    temps_p50K = temps + 50.0
    temps_m50K = temps - 50.0

    x_mid, y_mid, z_mid = example['x_mid'], example['y_mid'], example['z_mid']
    r_x, r_y, r_z = example['r_x'], example['r_y'], example['r_z']
    expected_vol = (4.0/3.0)*3.14 * r_x * r_y * r_z
    int_nomask = 0
    ext_mask = 1

    x, y, z, blank_vol = dm.define_grid_3d(Nx, Ny, Nz, Lx, Ly, Lz)

    mask = dm.set_grid_values_3d_rounded(x, y, z, blank_vol,
                                         x_mid, y_mid, z_mid,
                                         r_x, r_y, r_z,
                                         int_nomask, ext_mask)

    no_3mnths = iterations[0]
    no_4years = iterations[1]
    no_8years = iterations[2]
    no_10years = iterations[3]

    # first two
    # after 3 months and 10 years
    # rounding of olivine grains - want approx 50% rounded
    # at 3 months > 1623.15 OR at 10 years > 1573.15

    filename = f"{results_folder}/{run_no}/{run_no}_{no_3mnths}.npy"
    results = np.load(filename)
    masked_result_a = ma.masked_array(results, mask=mask)
    total_size = (masked_result_a > 100).sum()
    actual_volume = total_size * d_x * d_x * d_x
    mean_it0 = masked_result_a.mean()

    filename = f"{results_folder}/{run_no}/{run_no}_{no_10years}.npy"
    results = np.load(filename)
    masked_result_b = ma.masked_array(results, mask=mask)
    mean_it3 = masked_result_b.mean()

    rounding = np.bitwise_or(masked_result_a > temps[0], masked_result_b > temps[1]).sum()
    percent_rounded = 100*rounding/total_size

    rounding_p10pc = np.bitwise_or(masked_result_a > temps_p10pc[0], masked_result_b > temps_p10pc[1]).sum()
    percent_rounded_p10pc = 100*rounding_p10pc/total_size

    rounding_m10pc = np.bitwise_or(masked_result_a > temps_m10pc[0], masked_result_b > temps_m10pc[1]).sum()
    percent_rounded_m10pc = 100*rounding_m10pc/total_size

    rounding_p50K = np.bitwise_or(masked_result_a > temps_p50K[0], masked_result_b > temps_p50K[1]).sum()
    percent_rounded_p50K = 100*rounding_p50K/total_size

    rounding_m50K = np.bitwise_or(masked_result_a > temps_m50K[0], masked_result_b > temps_m50K[1]).sum()
    percent_rounded_m50K = 100*rounding_m50K/total_size

    print(percent_rounded)

    # Second two
    # after 4 years and 8 years
    # preservation of chemical zoning
    # at 4 years < 1573.15 AND at 8 years < 1373.15
    # Want at least 20 pc

    filename = f"{results_folder}/{run_no}/{run_no}_{no_4years}.npy"
    results = np.load(filename)
    masked_result_a = ma.masked_array(results, mask=mask)
    total_size = (masked_result_a>100).sum()
    mean_it1 = masked_result_a.mean()

    filename = f"{results_folder}/{run_no}/{run_no}_{no_8years}.npy"
    results = np.load(filename)
    masked_result_b = ma.masked_array(results, mask=mask)
    mean_it2 = masked_result_b.mean()

    geochem = np.bitwise_and(masked_result_a < temps[1], masked_result_b < temps[2]).sum()
    percent_geochem_preserved = 100*geochem/total_size

    geochem_p10pc = np.bitwise_and(masked_result_a < temps_p10pc[1], masked_result_b < temps_p10pc[2]).sum()
    percent_geochem_preserved_p10pc = 100*geochem_p10pc/total_size

    geochem_m10pc = np.bitwise_and(masked_result_a < temps_m10pc[1], masked_result_b < temps_m10pc[2]).sum()
    percent_geochem_preserved_m10pc = 100*geochem_m10pc/total_size

    geochem_p50K = np.bitwise_and(masked_result_a < temps_p50K[1], masked_result_b < temps_p50K[2]).sum()
    percent_geochem_preserved_p50K = 100*geochem_p50K/total_size

    geochem_m50K = np.bitwise_and(masked_result_a < temps_m50K[1], masked_result_b < temps_m50K[2]).sum()
    percent_geochem_preserved_m50K = 100*geochem_m50K/total_size

    print(f"Percent geochem preserved: {percent_geochem_preserved}")
    print(f"Percent rounded: {percent_rounded}")
    print(f"r_x: {r_x}; r_y: {r_y}; r_z = {r_z}")
    print(f"Expected volume: {expected_vol}; actual volume: {actual_volume}; difference: {actual_volume - expected_vol}")
    return (run_no,
            percent_rounded,
            percent_geochem_preserved,
            r_x, r_y,
            expected_vol,
            actual_volume,
            mean_it0,
            mean_it1,
            mean_it2,
            mean_it3, percent_rounded_p10pc, percent_rounded_m10pc, percent_rounded_p50K , percent_rounded_m50K,
           percent_geochem_preserved_p10pc, percent_geochem_preserved_m10pc, percent_geochem_preserved_p50K , percent_geochem_preserved_m50K)



def filtering_expand_bounds(job_no, params, results_folder, iterations=[3, 48, 96, 120]):
    # Expand or contract bounds by a certain amount
    # ignore this, does the same as previous function
    example = params.iloc[job_no]
    run_no = example['id']
    timestep = example['dt']
    Nx = example['Nx']
    Ny = example['Ny']
    Nz = example['Nz']

    Lx = example['Lx']
    Ly = example['Ly']
    Lz = example['Lz']

    d_x = Lx/(Nx - 1)

    temps = np.array([1623.15, 1573.15, 1373.15])
    temps_p10pc = temps + (temps * 0.01)
    temps_m10pc = temps - (temps * 0.01)
    temps_p50K = temps + 50.0
    temps_m50K = temps - 50.0

    x_mid, y_mid, z_mid = example['x_mid'], example['y_mid'], example['z_mid']
    r_x, r_y, r_z = example['r_x'], example['r_y'], example['r_z']
    expected_vol = (4.0/3.0)*3.14 * r_x * r_y * r_z
    int_nomask = 0
    ext_mask = 1

    x, y, z, blank_vol = dm.define_grid_3d(Nx, Ny, Nz, Lx, Ly, Lz)

    mask = dm.set_grid_values_3d_rounded(x, y, z, blank_vol,
                                         x_mid, y_mid, z_mid,
                                         r_x, r_y, r_z,
                                         int_nomask, ext_mask)

    no_3mnths = iterations[0]
    no_4years = iterations[1]
    no_8years = iterations[2]
    no_10years = iterations[3]

    # first two
    # after 3 months and 10 years
    # rounding of olivine grains - want approx 50% rounded
    # at 3 months > 1623.15 OR at 10 years > 1573.15

    filename = f"{results_folder}/{run_no}/{run_no}_{no_3mnths}.npy"
    results = np.load(filename)
    masked_result_a = ma.masked_array(results, mask=mask)
    total_size = (masked_result_a > 100).sum()
    actual_volume = total_size * d_x * d_x * d_x
    mean_it0 = masked_result_a.mean()

    filename = f"{results_folder}/{run_no}/{run_no}_{no_10years}.npy"
    results = np.load(filename)
    masked_result_b = ma.masked_array(results, mask=mask)
    mean_it3 = masked_result_b.mean()

    rounding = np.bitwise_or(masked_result_a > temps[0], masked_result_b > temps[1]).sum()
    percent_rounded = 100*rounding/total_size

    # Expand bounds by 10 pc
    rounding_p10pc = np.bitwise_or(masked_result_a > temps_p10pc[0], masked_result_b > temps_p10pc[1]).sum()
    percent_rounded_p10pc = 100*rounding_p10pc/total_size

    rounding_m10pc = np.bitwise_or(masked_result_a > temps_m10pc[0], masked_result_b > temps_m10pc[1]).sum()
    percent_rounded_m10pc = 100*rounding_m10pc/total_size

    rounding_p50K = np.bitwise_or(masked_result_a > temps_p50K[0], masked_result_b > temps_p50K[1]).sum()
    percent_rounded_p50K = 100*rounding_p50K/total_size

    rounding_m50K = np.bitwise_or(masked_result_a > temps_m50K[0], masked_result_b > temps_m50K[1]).sum()
    percent_rounded_m50K = 100*rounding_m50K/total_size

    print(percent_rounded)

    # Second two
    # after 4 years and 8 years
    # preservation of chemical zoning
    # at 4 years < 1573.15 AND at 8 years < 1373.15
    # Want at least 20 pc

    filename = f"{results_folder}/{run_no}/{run_no}_{no_4years}.npy"
    results = np.load(filename)
    masked_result_a = ma.masked_array(results, mask=mask)
    total_size = (masked_result_a>100).sum()
    mean_it1 = masked_result_a.mean()

    filename = f"{results_folder}/{run_no}/{run_no}_{no_8years}.npy"
    results = np.load(filename)
    masked_result_b = ma.masked_array(results, mask=mask)
    mean_it2 = masked_result_b.mean()

    geochem = np.bitwise_and(masked_result_a < temps[1], masked_result_b < temps[2]).sum()
    percent_geochem_preserved = 100*geochem/total_size

    geochem_p10pc = np.bitwise_and(masked_result_a < temps_p10pc[1], masked_result_b < temps_p10pc[2]).sum()
    percent_geochem_preserved_p10pc = 100*geochem_p10pc/total_size

    geochem_m10pc = np.bitwise_and(masked_result_a < temps_m10pc[1], masked_result_b < temps_m10pc[2]).sum()
    percent_geochem_preserved_m10pc = 100*geochem_m10pc/total_size

    geochem_p50K = np.bitwise_and(masked_result_a < temps_p50K[1], masked_result_b < temps_p50K[2]).sum()
    percent_geochem_preserved_p50K = 100*geochem_p50K/total_size

    geochem_m50K = np.bitwise_and(masked_result_a < temps_m50K[1], masked_result_b < temps_m50K[2]).sum()
    percent_geochem_preserved_m50K = 100*geochem_m50K/total_size

    print(f"Percent geochem preserved: {percent_geochem_preserved}")
    print(f"Percent rounded: {percent_rounded}")
    print(f"r_x: {r_x}; r_y: {r_y}; r_z = {r_z}")
    print(f"Expected volume: {expected_vol}; actual volume: {actual_volume}; difference: {actual_volume - expected_vol}")
    return (run_no,
            percent_rounded,
            percent_geochem_preserved,
            r_x, r_y,
            expected_vol,
            actual_volume,
            mean_it0,
            mean_it1,
            mean_it2,
            mean_it3, percent_rounded_p10pc, percent_rounded_m10pc, percent_rounded_p50K , percent_rounded_m50K,
           percent_geochem_preserved_p10pc, percent_geochem_preserved_m10pc, percent_geochem_preserved_p50K , percent_geochem_preserved_m50K)


def param_dict_to_df_checking_bounds(params, param_dict):
    # define column names
    column_names = ["model_id", "percent_rounded",
                    "percent_geochem_preserved",
                    "r_x", "r_y",
                    "expected_vol", "actual_volume",
                    "mean_it0", "mean_it1", "mean_it2", "mean_it3",
                    "percent_rounded_p10pc",
                    "percent_rounded_m10pc",
                    "percent_rounded_p50K",
                    "percent_rounded_m50K",
                    "percent_geochem_preserved_p10pc",
                    "percent_geochem_preserved_m10pc",
                    "percent_geochem_preserved_p50K",
                    "percent_geochem_preserved_m50K"]
    # turn the dict into a dataframe:
    df = pd.DataFrame.from_dict(param_dict, orient="index", columns=column_names)
    # copy the old param df so we can modify it
    new_params = params.copy(deep=True)
    # delete the  example row 1
    new_params.drop(index=new_params.index[0], axis=0, inplace=True)
    # lcean up the index so it matches our new dataframe
    new_params.reset_index(inplace=True)
    # this adds the new dataframe alongside the old, keeping values, making new cols
    new_params[list(df.columns)] = df.to_numpy()
    # some params may be repeated, delete these
    new_params.T.drop_duplicates().T
    # return the new appended dataframe, to be plotted/saved etc
    return new_params