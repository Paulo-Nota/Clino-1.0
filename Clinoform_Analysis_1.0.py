# ------------------------ Part 1: Imports, Logging, and Calculation Functions ------------------------ 

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
from scipy.stats import t
from matplotlib.gridspec import GridSpec
import threading
import logging
import matplotlib.style as mpl_style
import os
import re  # For regex in clinoform detection
from scipy.signal import argrelextrema  # For automatic rollover detection

# Set up logging to file and console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("clinoform_analysis.log"),
        logging.StreamHandler()
    ]
)

# ------------------------ Calculation Functions ------------------------

def calculate_rmse(y_actual, y_pred):
    """Calculate Root Mean Square Error."""
    return np.sqrt(np.mean((y_actual - y_pred) ** 2))

def calculate_r_squared(y_actual, y_pred):
    """Calculate R-squared."""
    ss_res = np.sum((y_actual - y_pred) ** 2)
    ss_tot = np.sum((y_actual - np.mean(y_actual)) ** 2)
    return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

def calculate_confidence_intervals(popt, pcov, n_data, alpha=0.05):
    """Calculate confidence intervals for fitted parameters."""
    n_params = len(popt)
    dof = max(0, n_data - n_params)
    t_val = t.ppf(1.0 - alpha / 2., dof) if dof > 0 else 0
    param_errors = np.sqrt(np.diag(pcov))
    ci = param_errors * t_val
    return ci

def validate_numeric_column(df, column_name):
    """Validate that a column contains numeric data."""
    if column_name not in df.columns:
        logging.error(f"Column '{column_name}' does not exist.")
        return False
    if not pd.api.types.is_numeric_dtype(df[column_name]):
        logging.error(f"Column '{column_name}' is not numeric.")
        return False
    if df[column_name].isnull().all():
        logging.error(f"Column '{column_name}' is empty.")
        return False
    return True

def numerical_second_derivative(y, x):
    """Calculate numerical second derivative."""
    dy = np.gradient(y, x)
    d2y = np.gradient(dy, x)
    return d2y

def calculate_concavity_numerical(y, x):
    """Calculate concavity using numerical second derivatives."""
    d2y = numerical_second_derivative(y, x)
    return np.mean(d2y)

def calculate_dip(fit_func, x_vals, popt):
    """Calculate dip (slope) at the end of the foreset."""
    y_vals = fit_func(x_vals, *popt)
    slopes = np.gradient(y_vals, x_vals)
    return slopes[-1]

def calculate_slope(fit_func, x_vals, popt):
    """Calculate average slope (gradient)."""
    y_vals = fit_func(x_vals, *popt)
    slopes = np.gradient(y_vals, x_vals)
    return np.mean(slopes)

# ------------------------ Part 2: Data Processing and Analysis Functions ------------------------

# Function to perform curve fitting and return relevant metrics
def fit_model(fit_func, x_data, y_data, p0=None, bounds=(-np.inf, np.inf), maxfev=10000):
    try:
        popt, pcov = curve_fit(fit_func, x_data, y_data, p0=p0, bounds=bounds, maxfev=maxfev)
        y_pred = fit_func(x_data, *popt)
        r_squared = calculate_r_squared(y_data, y_pred)
        rmse = calculate_rmse(y_data, y_pred)
        ci = calculate_confidence_intervals(popt, pcov, len(x_data))
        return popt, pcov, r_squared, rmse, ci, y_pred
    except Exception as e:
        logging.warning(f"Curve fitting failed for {fit_func.__name__}: {e}")
        return None, None, None, None, None, None

# Function to detect clinoforms dynamically based on column naming patterns
def detect_clinoforms(df, x_prefix="X", y_prefix="Y", rollover_prefix="Rollover"):
    """
    Detects clinoforms in the DataFrame based on column naming patterns.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        x_prefix (str): Prefix for X columns.
        y_prefix (str): Prefix for Y columns.
        rollover_prefix (str): Prefix for Rollover columns.

    Returns:
        list of tuples: Each tuple contains (clinoform_name, (X_col, Y_col, Rollover_col))
    """
    clinoforms = []
    # Regex pattern to match X, Y, Rollover columns with optional numbers
    pattern = re.compile(rf'^({x_prefix}|{y_prefix}|{rollover_prefix})(\d*)\b', re.IGNORECASE)

    # Extract all column names
    columns = df.columns.tolist()

    # Initialize a dictionary to hold clinoform components
    clinoform_dict = {}

    for col in columns:
        match = pattern.match(col.strip())
        if match:
            prefix = match.group(1)
            number = match.group(2) if match.group(2) else '1'  # Default to '1' if no number
            clinoform_name = f"Clinoform {number}"
            if clinoform_name not in clinoform_dict:
                clinoform_dict[clinoform_name] = {'X': None, 'Y': None, 'Rollover': None}
            if prefix.lower() == x_prefix.lower():
                clinoform_dict[clinoform_name]['X'] = col.strip()
            elif prefix.lower() == y_prefix.lower():
                clinoform_dict[clinoform_name]['Y'] = col.strip()
            elif prefix.lower() == rollover_prefix.lower():
                clinoform_dict[clinoform_name]['Rollover'] = col.strip()

    # Convert the dictionary to a list of tuples
    for name, cols in clinoform_dict.items():
        if cols['X'] and cols['Y']:  # Rollover can be None
            clinoforms.append((name, (cols['X'], cols['Y'], cols['Rollover'])))
        else:
            logging.warning(f"Incomplete columns for {name}. Expected at least X and Y. Skipping this clinoform.")

    return clinoforms

# Function to automatically detect rollovers if not provided
def automatic_rollover_detection(x_data, y_data):
    """
    Automatically detects rollover points in the clinoform data.

    Parameters:
        x_data (np.ndarray): The X data.
        y_data (np.ndarray): The Y data.

    Returns:
        list: Indices of detected rollover points.
    """
    try:
        # Calculate second derivative (curvature)
        curvature = numerical_second_derivative(y_data, x_data)
        curvature_abs = np.abs(curvature)

        # Identify local maxima in curvature
        rollover_indices = argrelextrema(curvature_abs, np.greater)[0]

        # If more than two rollover points are found, select the two with the highest curvature
        if len(rollover_indices) > 2:
            sorted_indices = np.argsort(curvature_abs[rollover_indices])[::-1]
            rollover_indices = rollover_indices[sorted_indices[:2]]
            rollover_indices.sort()
        elif len(rollover_indices) < 2:
            logging.warning("Less than two rollover points detected.")
            # Return whatever is found

        return rollover_indices.tolist()
    except Exception as e:
        logging.error(f"Automatic rollover detection failed: {e}")
        return []

# Function to perform the analysis for each clinoform
def analyze_clinoform(df, x_col, y_col, rollover_col, models, clinoform_number=1,
                      rollover_option='auto'):
    # Validate x_col and y_col
    if not all(validate_numeric_column(df, col) for col in [x_col, y_col]):
        logging.error(f"Invalid data in {x_col} or {y_col}.")
        return None

    # Clean DataFrame
    df_clean = df[[x_col, y_col]].dropna(subset=[x_col, y_col])

    x_data = df_clean[x_col].to_numpy()
    y_data = df_clean[y_col].to_numpy()

    # Handle rollovers based on user's choice
    rollover_indices = []
    if rollover_col and rollover_col in df.columns and rollover_option != 'no_rollover':
        df_clean[rollover_col] = df[rollover_col]
        rollover_indices = df_clean[df_clean[rollover_col].notna()].index.tolist()
        rollover_indices = [i - df_clean.index[0] for i in rollover_indices]

        if len(rollover_indices) < 2:
            logging.warning("Less than two rollover points provided.")
            if rollover_option == 'auto':
                rollover_indices = automatic_rollover_detection(x_data, y_data)
            elif rollover_option == 'no_rollover':
                rollover_indices = []
    else:
        if rollover_option == 'auto':
            rollover_indices = automatic_rollover_detection(x_data, y_data)
        elif rollover_option == 'no_rollover':
            rollover_indices = []

    num_rollovers = len(rollover_indices)
    logging.info(f"Number of rollover points detected: {num_rollovers}")

    if num_rollovers >= 2:
        # Proceed with topset, foreset, and bottomset
        roll1_index = rollover_indices[0]
        roll2_index = rollover_indices[1]

        x_topset = x_data[:roll1_index + 1]
        y_topset = y_data[:roll1_index + 1]

        x_foreset = x_data[roll1_index:roll2_index + 1]
        y_foreset = y_data[roll1_index:roll2_index + 1]

        x_bottomset = x_data[roll2_index:]
        y_bottomset = y_data[roll2_index:]

        segment_info = 'Three segments (topset, foreset, bottomset)'
    elif num_rollovers == 1:
        # Proceed with two segments
        roll1_index = rollover_indices[0]

        x_topset = x_data[:roll1_index + 1]
        y_topset = y_data[:roll1_index + 1]

        x_foreset = x_data[roll1_index:]
        y_foreset = y_data[roll1_index:]

        x_bottomset = np.array([])
        y_bottomset = np.array([])

        segment_info = 'Two segments (topset and foreset)'
    else:
        # Proceed with entire clinoform as one segment
        x_topset = np.array([])
        y_topset = np.array([])

        x_foreset = x_data
        y_foreset = y_data

        x_bottomset = np.array([])
        y_bottomset = np.array([])

        segment_info = 'One segment (entire clinoform)'

    logging.info(f"Processing {segment_info} for Clinoform {clinoform_number}")

    # For Exponential and Gaussian fits, normalize x_foreset and shift y_foreset
    x_min = np.min(x_foreset)
    x_range = np.ptp(x_foreset) if np.ptp(x_foreset) != 0 else 1
    x_foreset_norm = (x_foreset - x_min) / x_range
    y_min = np.min(y_foreset)
    y_foreset_shifted = y_foreset - y_min

    results_dict = {
        'Fitting Model': [],
        'Equation': [],
        'R²': [],
        'RMSE': [],
        'Confidence Intervals': [],
        'Slope (Gradient)': [],
        'Topset Height': [],
        'Topset Length': [],
        'Foreset Height': [],
        'Foreset Length': [],
        'Clinoform Height': [],
        'Clinoform Width': [],
        'Upper Rollover': [],
        'Lower Rollover': [],
        'Concavity (Topset)': [],
        'Concavity (Foreset)': [],
        'Foreset Dip': []
    }

    # Additional Calculations
    topset_height = np.max(y_topset) - np.min(y_topset) if len(y_topset) > 0 else "N/A"
    topset_length = np.max(x_topset) - np.min(x_topset) if len(x_topset) > 0 else "N/A"

    foreset_height = np.max(y_foreset) - np.min(y_foreset)
    foreset_length = np.max(x_foreset) - np.min(x_foreset)

    clinoform_height = np.max(y_data) - np.min(y_data)
    clinoform_width = np.max(x_data) - np.min(x_data)

    upper_rollover = x_data[rollover_indices[0]] if num_rollovers >= 1 else "N/A"
    lower_rollover = x_data[rollover_indices[1]] if num_rollovers >= 2 else "N/A"

    # Prepare data for plotting
    plot_data = {
        'x_data': x_data,
        'y_data': y_data,
        'x_topset': x_topset,
        'y_topset': y_topset,
        'x_foreset': x_foreset,
        'y_foreset': y_foreset,
        'x_bottomset': x_bottomset,
        'y_bottomset': y_bottomset,
        'rollover_indices': rollover_indices,
        'models': {},
        'segment_label': 'Foreset' if num_rollovers > 0 else 'Entire Clinoform',
        'x_min': x_min,
        'x_range': x_range,
        'y_min': y_min,
        'x_foreset_norm': x_foreset_norm,
        'y_foreset_shifted': y_foreset_shifted,
        'clinoform_number': clinoform_number,
        'x_col': x_col,
        'y_col': y_col
    }

    # Topset: Linear, Inverse Quadratic, and Quadratic Fit
    if len(x_topset) > 0 and len(y_topset) > 0:
        if 'Linear' in models:
            # Linear Fit
            linear_model_topset = LinearRegression()
            linear_model_topset.fit(x_topset.reshape(-1, 1), y_topset)
            slope_topset = linear_model_topset.coef_[0]
            intercept_topset = linear_model_topset.intercept_
            linear_pred_topset = linear_model_topset.predict(x_topset.reshape(-1, 1))
            r_squared_linear_topset = calculate_r_squared(y_topset, linear_pred_topset)
            rmse_linear_topset = calculate_rmse(y_topset, linear_pred_topset)
            equation_linear_topset = f"y = {slope_topset:.4f}x + {intercept_topset:.4f}"

            # Calculate slope and other metrics
            avg_slope_topset = slope_topset
            concavity_topset = 0  # Linear model has zero concavity

            # Store results
            results_dict['Fitting Model'].append('Topset Linear')
            results_dict['Equation'].append(equation_linear_topset)
            results_dict['R²'].append(r_squared_linear_topset)
            results_dict['RMSE'].append(rmse_linear_topset)
            results_dict['Confidence Intervals'].append('N/A')
            results_dict['Slope (Gradient)'].append(avg_slope_topset)
            results_dict['Topset Height'].append(topset_height)
            results_dict['Topset Length'].append(topset_length)
            results_dict['Foreset Height'].append(foreset_height)
            results_dict['Foreset Length'].append(foreset_length)
            results_dict['Clinoform Height'].append(clinoform_height)
            results_dict['Clinoform Width'].append(clinoform_width)
            results_dict['Upper Rollover'].append(upper_rollover)
            results_dict['Lower Rollover'].append(lower_rollover)
            results_dict['Concavity (Topset)'].append(concavity_topset)
            results_dict['Concavity (Foreset)'].append("N/A")
            results_dict['Foreset Dip'].append("N/A")

            # Store for plotting
            plot_data['models']['Topset Linear'] = {
                'x': x_topset,
                'y': linear_pred_topset,
                'label': f'Topset Linear Fit (R²={r_squared_linear_topset:.2f})',
                'style': {'color': 'green'}
            }

        if 'Inverse Quadratic' in models:
            # Inverse Quadratic Fit
            def inverse_quadratic(x, a, b, c):
                return a + b * x + c / (x ** 2 + 1e-8)  # Add small number to prevent division by zero

            popt_quad_inv, pcov_quad_inv, r_squared_quad_inv, rmse_quad_inv, ci_quad_inv, inv_quad_pred = fit_model(
                inverse_quadratic, x_topset, y_topset)
            equation_inverse_quadratic_topset = f"y = {popt_quad_inv[0]:.4f} + {popt_quad_inv[1]:.4f}x + {popt_quad_inv[2]:.4f}/x²" if popt_quad_inv is not None else "Fit failed"

            if popt_quad_inv is not None:
                slope_inv_quad = calculate_slope(inverse_quadratic, x_topset, popt_quad_inv)
                concavity_inv_quad = calculate_concavity_numerical(inv_quad_pred, x_topset)
                # Store for plotting
                plot_data['models']['Topset Inverse Quadratic'] = {
                    'x': x_topset,
                    'y': inv_quad_pred,
                    'label': f'Topset Inverse Quadratic Fit (R²={r_squared_quad_inv:.2f})',
                    'style': {'color': 'cyan', 'linestyle': '--'}
                }
            else:
                slope_inv_quad = "N/A"
                concavity_inv_quad = "N/A"

            # Store results
            results_dict['Fitting Model'].append('Topset Inverse Quadratic')
            results_dict['Equation'].append(equation_inverse_quadratic_topset)
            results_dict['R²'].append(r_squared_quad_inv)
            results_dict['RMSE'].append(rmse_quad_inv)
            results_dict['Confidence Intervals'].append(ci_quad_inv.tolist() if ci_quad_inv is not None else 'N/A')
            results_dict['Slope (Gradient)'].append(slope_inv_quad)
            results_dict['Topset Height'].append(topset_height)
            results_dict['Topset Length'].append(topset_length)
            results_dict['Foreset Height'].append(foreset_height)
            results_dict['Foreset Length'].append(foreset_length)
            results_dict['Clinoform Height'].append(clinoform_height)
            results_dict['Clinoform Width'].append(clinoform_width)
            results_dict['Upper Rollover'].append(upper_rollover)
            results_dict['Lower Rollover'].append(lower_rollover)
            results_dict['Concavity (Topset)'].append(concavity_inv_quad)
            results_dict['Concavity (Foreset)'].append("N/A")
            results_dict['Foreset Dip'].append("N/A")

        if 'Quadratic' in models:
            # Quadratic Fit
            def quadratic_model(x, a, b, c):
                return a * x ** 2 + b * x + c

            popt_quad_topset, pcov_quad_topset, r_squared_quad_topset, rmse_quad_topset, ci_quad_topset, quad_pred_topset = fit_model(
                quadratic_model, x_topset, y_topset)
            equation_quadratic_topset = f"y = {popt_quad_topset[0]:.4f}x² + {popt_quad_topset[1]:.4f}x + {popt_quad_topset[2]:.4f}" if popt_quad_topset is not None else "Fit failed"

            if popt_quad_topset is not None:
                slope_quad_topset = calculate_slope(quadratic_model, x_topset, popt_quad_topset)
                concavity_quad_topset = calculate_concavity_numerical(quad_pred_topset, x_topset)
                # Store for plotting
                plot_data['models']['Topset Quadratic'] = {
                    'x': x_topset,
                    'y': quad_pred_topset,
                    'label': f'Topset Quadratic Fit (R²={r_squared_quad_topset:.2f})',
                    'style': {'color': 'orange', 'linestyle': ':'}
                }
            else:
                slope_quad_topset = "N/A"
                concavity_quad_topset = "N/A"

            # Store results
            results_dict['Fitting Model'].append('Topset Quadratic')
            results_dict['Equation'].append(equation_quadratic_topset)
            results_dict['R²'].append(r_squared_quad_topset)
            results_dict['RMSE'].append(rmse_quad_topset)
            results_dict['Confidence Intervals'].append(ci_quad_topset.tolist() if ci_quad_topset is not None else 'N/A')
            results_dict['Slope (Gradient)'].append(slope_quad_topset)
            results_dict['Topset Height'].append(topset_height)
            results_dict['Topset Length'].append(topset_length)
            results_dict['Foreset Height'].append(foreset_height)
            results_dict['Foreset Length'].append(foreset_length)
            results_dict['Clinoform Height'].append(clinoform_height)
            results_dict['Clinoform Width'].append(clinoform_width)
            results_dict['Upper Rollover'].append(upper_rollover)
            results_dict['Lower Rollover'].append(lower_rollover)
            results_dict['Concavity (Topset)'].append(concavity_quad_topset)
            results_dict['Concavity (Foreset)'].append("N/A")
            results_dict['Foreset Dip'].append("N/A")

    # Foreset or Entire Clinoform: Exponential, Gaussian, and Quadratic Fit
    if 'Exponential' in models:
        # Exponential Fit
        def exponential_model(x, A, k):
            return A * np.exp(k * x)

        popt_exp, pcov_exp, r_squared_exp_foreset, rmse_exp_foreset, ci_exp, exp_pred_norm = fit_model(
            exponential_model, x_foreset_norm, y_foreset_shifted, p0=(1, -1))
        equation_exponential_foreset = f"y = {popt_exp[0]:.4f} * exp({popt_exp[1]:.4f}x)" if popt_exp is not None else "Fit failed"

        if popt_exp is not None:
            # Create adjusted exponential model to use original x values
            def exponential_model_original_scale(x, A, k):
                x_norm = (x - x_min) / x_range
                return A * np.exp(k * x_norm)

            exp_pred_foreset = exponential_model_original_scale(x_foreset, *popt_exp) + y_min
            # Calculate concavity numerically
            foreset_concavity_exp = calculate_concavity_numerical(exp_pred_foreset, x_foreset)
            foreset_dip_exp = calculate_dip(exponential_model_original_scale, x_foreset, popt_exp)
            # Store for plotting
            plot_data['models'][f'{plot_data["segment_label"]} Exponential'] = {
                'x': x_foreset,
                'y': exp_pred_foreset,
                'label': f'{plot_data["segment_label"]} Exponential Fit (R²={r_squared_exp_foreset:.2f})',
                'style': {'color': 'red'}
            }
        else:
            foreset_concavity_exp = "N/A"
            foreset_dip_exp = "N/A"

        # Store results
        segment_label = 'Foreset' if num_rollovers > 0 else 'Entire Clinoform'
        results_dict['Fitting Model'].append(f'{segment_label} Exponential')
        results_dict['Equation'].append(equation_exponential_foreset)
        results_dict['R²'].append(r_squared_exp_foreset)
        results_dict['RMSE'].append(rmse_exp_foreset)
        results_dict['Confidence Intervals'].append(ci_exp.tolist() if ci_exp is not None else 'N/A')
        results_dict['Slope (Gradient)'].append(calculate_slope(exponential_model_original_scale, x_foreset, popt_exp) if popt_exp is not None else 'N/A')
        results_dict['Topset Height'].append(topset_height)
        results_dict['Topset Length'].append(topset_length)
        results_dict['Foreset Height'].append(foreset_height)
        results_dict['Foreset Length'].append(foreset_length)
        results_dict['Clinoform Height'].append(clinoform_height)
        results_dict['Clinoform Width'].append(clinoform_width)
        results_dict['Upper Rollover'].append(upper_rollover)
        results_dict['Lower Rollover'].append(lower_rollover)
        results_dict['Concavity (Topset)'].append("N/A")
        results_dict['Concavity (Foreset)'].append(foreset_concavity_exp)
        results_dict['Foreset Dip'].append(foreset_dip_exp)

    if 'Gaussian' in models:
        # Gaussian Fit
        def gaussian_model(x, A, mu, sigma):
            return A * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2 + 1e-8))

        popt_gauss, pcov_gauss, r_squared_gauss_foreset, rmse_gauss_foreset, ci_gauss, gauss_pred_norm = fit_model(
            gaussian_model, x_foreset_norm, y_foreset_shifted, p0=(np.max(y_foreset_shifted), 0.5, 0.1))

        if popt_gauss is not None:
            # Create adjusted gaussian model to use original x values
            def gaussian_model_original_scale(x, A, mu, sigma):
                x_norm = (x - x_min) / x_range
                return A * np.exp(-((x_norm - mu) ** 2) / (2 * sigma ** 2 + 1e-8))

            gauss_pred_foreset = gaussian_model_original_scale(x_foreset, *popt_gauss) + y_min
            equation_gaussian_foreset = f"y = {popt_gauss[0]:.4f} * exp(-((x - {popt_gauss[1]:.4f})² / (2 * {popt_gauss[2]:.4f}²)))"
            # Calculate concavity numerically
            foreset_concavity_gauss = calculate_concavity_numerical(gauss_pred_foreset, x_foreset)
            foreset_dip_gauss = calculate_dip(gaussian_model_original_scale, x_foreset, popt_gauss)
            # Store for plotting
            plot_data['models'][f'{plot_data["segment_label"]} Gaussian'] = {
                'x': x_foreset,
                'y': gauss_pred_foreset,
                'label': f'{plot_data["segment_label"]} Gaussian Fit (R²={r_squared_gauss_foreset:.2f})',
                'style': {'color': 'purple', 'linestyle': '--'}
            }
        else:
            gauss_pred_foreset = None
            equation_gaussian_foreset = "Fit failed"
            foreset_concavity_gauss = "N/A"
            foreset_dip_gauss = "N/A"

        # Store results
        segment_label = 'Foreset' if num_rollovers > 0 else 'Entire Clinoform'
        results_dict['Fitting Model'].append(f'{segment_label} Gaussian')
        results_dict['Equation'].append(equation_gaussian_foreset)
        results_dict['R²'].append(r_squared_gauss_foreset)
        results_dict['RMSE'].append(rmse_gauss_foreset)
        results_dict['Confidence Intervals'].append(ci_gauss.tolist() if ci_gauss is not None else 'N/A')
        results_dict['Slope (Gradient)'].append(calculate_slope(gaussian_model_original_scale, x_foreset, popt_gauss) if popt_gauss is not None else 'N/A')
        results_dict['Topset Height'].append(topset_height)
        results_dict['Topset Length'].append(topset_length)
        results_dict['Foreset Height'].append(foreset_height)
        results_dict['Foreset Length'].append(foreset_length)
        results_dict['Clinoform Height'].append(clinoform_height)
        results_dict['Clinoform Width'].append(clinoform_width)
        results_dict['Upper Rollover'].append(upper_rollover)
        results_dict['Lower Rollover'].append(lower_rollover)
        results_dict['Concavity (Topset)'].append("N/A")
        results_dict['Concavity (Foreset)'].append(foreset_concavity_gauss)
        results_dict['Foreset Dip'].append(foreset_dip_gauss)

    if 'Quadratic' in models:
        # Quadratic Fit (for Foreset or Entire Clinoform)
        def quadratic_model(x, a, b, c):
            return a * x ** 2 + b * x + c

        popt_quad_foreset, pcov_quad_foreset, r_squared_quad_foreset, rmse_quad_foreset, ci_quad_foreset, quad_pred_foreset = fit_model(
            quadratic_model, x_foreset, y_foreset)

        if popt_quad_foreset is not None:
            equation_quadratic_foreset = f"y = {popt_quad_foreset[0]:.4f}x² + {popt_quad_foreset[1]:.4f}x + {popt_quad_foreset[2]:.4f}"
            concavity_quad_foreset = calculate_concavity_numerical(quad_pred_foreset, x_foreset)
            foreset_dip_quad = calculate_dip(quadratic_model, x_foreset, popt_quad_foreset)
            # Store for plotting
            plot_data['models'][f'{plot_data["segment_label"]} Quadratic'] = {
                'x': x_foreset,
                'y': quad_pred_foreset,
                'label': f'{plot_data["segment_label"]} Quadratic Fit (R²={r_squared_quad_foreset:.2f})',
                'style': {'color': 'orange', 'linestyle': ':'}
            }
        else:
            quad_pred_foreset = None
            equation_quadratic_foreset = "Fit failed"
            concavity_quad_foreset = "N/A"
            foreset_dip_quad = "N/A"

        # Store results
        segment_label = 'Foreset' if num_rollovers > 0 else 'Entire Clinoform'
        results_dict['Fitting Model'].append(f'{segment_label} Quadratic')
        results_dict['Equation'].append(equation_quadratic_foreset)
        results_dict['R²'].append(r_squared_quad_foreset)
        results_dict['RMSE'].append(rmse_quad_foreset)
        results_dict['Confidence Intervals'].append(ci_quad_foreset.tolist() if ci_quad_foreset is not None else 'N/A')
        results_dict['Slope (Gradient)'].append(calculate_slope(quadratic_model, x_foreset, popt_quad_foreset) if popt_quad_foreset is not None else 'N/A')
        results_dict['Topset Height'].append(topset_height)
        results_dict['Topset Length'].append(topset_length)
        results_dict['Foreset Height'].append(foreset_height)
        results_dict['Foreset Length'].append(foreset_length)
        results_dict['Clinoform Height'].append(clinoform_height)
        results_dict['Clinoform Width'].append(clinoform_width)
        results_dict['Upper Rollover'].append(upper_rollover)
        results_dict['Lower Rollover'].append(lower_rollover)
        results_dict['Concavity (Topset)'].append("N/A")
        results_dict['Concavity (Foreset)'].append(concavity_quad_foreset)
        results_dict['Foreset Dip'].append(foreset_dip_quad)

    # Bottomset: Linear Fit
    if len(x_bottomset) > 0 and len(y_bottomset) > 0 and 'Linear' in models:
        # Linear Fit for Bottomset
        linear_model_bottomset = LinearRegression()
        linear_model_bottomset.fit(x_bottomset.reshape(-1, 1), y_bottomset)
        slope_bottomset = linear_model_bottomset.coef_[0]
        intercept_bottomset = linear_model_bottomset.intercept_
        linear_pred_bottomset = linear_model_bottomset.predict(x_bottomset.reshape(-1, 1))
        r_squared_linear_bottomset = calculate_r_squared(y_bottomset, linear_pred_bottomset)
        rmse_linear_bottomset = calculate_rmse(y_bottomset, linear_pred_bottomset)
        equation_linear_bottomset = f"y = {slope_bottomset:.4f}x + {intercept_bottomset:.4f}"

        # Store results
        results_dict['Fitting Model'].append('Bottomset Linear')
        results_dict['Equation'].append(equation_linear_bottomset)
        results_dict['R²'].append(r_squared_linear_bottomset)
        results_dict['RMSE'].append(rmse_linear_bottomset)
        results_dict['Confidence Intervals'].append('N/A')
        results_dict['Slope (Gradient)'].append(slope_bottomset)
        results_dict['Topset Height'].append(topset_height)
        results_dict['Topset Length'].append(topset_length)
        results_dict['Foreset Height'].append(foreset_height)
        results_dict['Foreset Length'].append(foreset_length)
        results_dict['Clinoform Height'].append(clinoform_height)
        results_dict['Clinoform Width'].append(clinoform_width)
        results_dict['Upper Rollover'].append(upper_rollover)
        results_dict['Lower Rollover'].append(lower_rollover)
        results_dict['Concavity (Topset)'].append("N/A")
        results_dict['Concavity (Foreset)'].append("N/A")
        results_dict['Foreset Dip'].append("N/A")

        # Store for plotting
        plot_data['models']['Bottomset Linear'] = {
            'x': x_bottomset,
            'y': linear_pred_bottomset,
            'label': f'Bottomset Linear Fit (R²={r_squared_linear_bottomset:.2f})',
            'style': {'color': 'magenta'}
        }

    return pd.DataFrame(results_dict), plot_data

# ------------------------ Part 3 and 4: GUI Setup, Function Definitions, and Main Execution ------------------------

import tkinter as tk
from tkinter import filedialog, messagebox, ttk, colorchooser, simpledialog
import threading
import logging
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import mplcursors

# Initialize Tkinter root
root = tk.Tk()
root.title("Clinoform Analysis")
root.geometry('1300x800')  # Set initial window size
root.minsize(800, 600)
root.rowconfigure(0, weight=1)
root.columnconfigure(0, weight=1)

# Global variables
clinoform_vars = {}
model_vars = {}
plot_data_dict = {}
plot_canvas_frame = None
plot_selection_listbox = None
plot_mode_var = None
show_legend_var = None
full_results_df = None  # To store the full results DataFrame

# Functions for GUI operations

def update_progress_bar(progress_bar, current, total):
    progress_value = (current / total) * 100
    progress_bar['value'] = progress_value
    progress_label.config(text=f"Processing Clinoform {current} of {total}")
    root.update_idletasks()

def open_file():
    file_path = filedialog.askopenfilename(title="Select Excel or CSV File", filetypes=[("Excel files", "*.xlsx *.xls"), ("CSV files", "*.csv")])
    if file_path:
        if not (file_path.endswith('.xlsx') or file_path.endswith('.xls') or file_path.endswith('.csv')):
            messagebox.showerror("Error", "Please select a valid Excel (.xlsx, .xls) or CSV (.csv) file.")
            return
        file_label.config(text=file_path)
        try:
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
                df.columns = df.columns.str.strip()  # Strip column names
                sheet_name_combobox['values'] = ['Sheet1']
                sheet_name_combobox.current(0)
                on_sheet_selected(None)  # Proceed to clinoform detection
            else:
                excel_file = pd.ExcelFile(file_path)
                sheet_names = excel_file.sheet_names
                sheet_names = [name.strip() for name in sheet_names]  # Strip sheet names
                sheet_name_combobox['values'] = sheet_names
                if len(sheet_names) > 0:
                    sheet_name_combobox.current(0)
        except Exception as e:
            messagebox.showerror("Error", f"Unable to load file: {e}")

def on_sheet_selected(event):
    sheet_name = sheet_name_combobox.get()
    file_path = file_label.cget("text")
    if not sheet_name or not file_path:
        return
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
            df.columns = df.columns.str.strip()  # Strip column names
        else:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            df.columns = df.columns.str.strip()  # Strip column names
        clinoforms = detect_clinoforms(df)

        # Log detected clinoforms
        logging.info(f"Detected Clinoforms: {clinoforms}")

        # Clear existing clinoform checkbuttons
        for widget in clinoform_frame.winfo_children():
            widget.destroy()

        # Display detected clinoforms
        if clinoforms:
            ttk.Label(clinoform_frame, text=f"Detected {len(clinoforms)} Clinoform(s):").grid(row=0, column=0, sticky="w", padx=5)
            for idx, (clinoform_name, (x_col, y_col, rollover_col)) in enumerate(clinoforms):
                var = tk.BooleanVar(value=True)  # Default to selected
                chk = ttk.Checkbutton(clinoform_frame, text=clinoform_name, variable=var)
                chk.grid(row=idx+1, column=0, sticky="w", padx=5, pady=2)
                clinoform_vars[clinoform_name] = (x_col, y_col, rollover_col, var)
        else:
            ttk.Label(clinoform_frame, text="No clinoforms detected.").grid(row=0, column=0, sticky="w", padx=5)
    except Exception as e:
        messagebox.showerror("Error", f"Unable to load sheet: {e}")

def reset_all():
    file_label.config(text="No file selected")
    sheet_name_combobox.set('')
    for clino, (x_col, y_col, rollover_col, var) in clinoform_vars.items():
        var.set(False)
    for model, var in model_vars.items():
        var.set(False)
    rollover_handling_var.set('auto')
    progress['value'] = 0
    progress_label.config(text="Progress")
    for row in tree.get_children():
        tree.delete(row)
    if plot_canvas_frame:
        for widget in plot_canvas_frame.winfo_children():
            widget.destroy()
    if plot_selection_listbox:
        plot_selection_listbox.delete(0, tk.END)
    logging.info("All selections have been reset.")

def analyze():
    selected_clinoforms = [name for name, (x_col, y_col, rollover_col, var) in clinoform_vars.items() if var.get()]
    selected_models = [model_name for model_name, var in model_vars.items() if var.get()]
    rollover_option = rollover_handling_var.get()
    if not selected_clinoforms:
        messagebox.showerror("Error", "No clinoforms selected.")
        return
    if not selected_models:
        messagebox.showerror("Error", "No fitting models selected.")
        return
    file_path = file_label.cget("text")
    sheet_name = sheet_name_combobox.get()
    if not file_path:
        messagebox.showerror("Error", "No file selected.")
        return

    # Prompt user to select output file
    output_file = filedialog.asksaveasfilename(defaultextension=".xlsx",
                                               filetypes=[("Excel Files", "*.xlsx")],
                                               title="Save Analysis Results As")
    if not output_file:
        messagebox.showinfo("Analysis Cancelled", "Analysis cancelled by user.")
        return

    threading.Thread(target=run_analysis, args=(file_path, sheet_name, selected_clinoforms, selected_models, rollover_option, output_file)).start()

def run_analysis(file_path, sheet_name, selected_clinoforms, selected_models, rollover_option, output_file):
    global plot_data_dict, full_results_df
    plot_data_dict = {}
    results = []
    total_clinoforms = len(selected_clinoforms)

    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
            df.columns = df.columns.str.strip()
        else:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            df.columns = df.columns.str.strip()

        # Build a mapping of clinoform names to their columns
        clinoform_mapping = detect_clinoforms(df)

        clinoform_dict = {name: (x_col, y_col, rollover_col) for name, (x_col, y_col, rollover_col) in clinoform_mapping}

        for idx, clinoform_name in enumerate(selected_clinoforms, 1):
            x_col, y_col, rollover_col = clinoform_dict[clinoform_name]
            analysis_result = analyze_clinoform(df, x_col, y_col, rollover_col, selected_models, clinoform_number=clinoform_name, rollover_option=rollover_option)
            if analysis_result:
                result_df, plot_data = analysis_result
                # Append clinoform name to the result DataFrame
                result_df.insert(0, 'Clinoform', clinoform_name)
                results.append(result_df)
                # Store plot data
                plot_data_dict[clinoform_name] = plot_data
            update_progress_bar(progress, idx, total_clinoforms)
        if results:
            full_results_df = pd.concat(results, ignore_index=True)
            display_results_summary(full_results_df)
            # Update plot selection listbox
            plot_selection_listbox.delete(0, tk.END)
            for clinoform_name in plot_data_dict.keys():
                plot_selection_listbox.insert(tk.END, clinoform_name)
            # Save results to user-specified Excel file
            save_results_to_excel(full_results_df, output_file)
            messagebox.showinfo("Analysis Complete", f"Clinoform analysis completed successfully.\nResults have been saved to '{output_file}'.")
        else:
            messagebox.showwarning("No Results", "No results to display.")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred during analysis: {e}")
        logging.error(f"An error occurred during analysis: {e}")

def save_results_to_excel(results_df, output_file):
    if results_df is not None:
        try:
            with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                # Group results by Clinoform and write each to a separate sheet
                for clinoform_name, group_df in results_df.groupby('Clinoform'):
                    sheet_name = clinoform_name.replace('/', '_')  # Replace any invalid characters
                    group_df.to_excel(writer, sheet_name=sheet_name[:31], index=False)  # Sheet name max length is 31
            logging.info(f"Results have been saved to '{output_file}'")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save results to Excel: {e}")
            logging.error(f"Failed to save results to Excel: {e}")
    else:
        messagebox.showwarning("No Results", "No results to save.")

def choose_color(color_var):
    color_code = colorchooser.askcolor(title="Choose Line Color")
    if color_code:
        color_var.set(color_code[1])
        update_plot_display()

def update_plot_display(event=None):
    selected_plots = [plot_selection_listbox.get(i) for i in plot_selection_listbox.curselection()]
    plot_mode = plot_mode_var.get()
    show_legend = show_legend_var.get()

    line_color = line_color_var.get()
    line_style = line_style_var.get()
    marker = marker_var.get()
    legend_loc = legend_loc_var.get()
    font_size = font_size_var.get()

    if not selected_plots:
        # Do not show error to avoid annoyance when changing options
        return
    if not plot_data_dict:
        messagebox.showerror("Error", "No plot data available.")
        return

    # Clear existing plots
    for widget in plot_canvas_frame.winfo_children():
        widget.destroy()

    # Create new plots
    if plot_mode == 'Overlay':
        fig, ax = plt.subplots(figsize=(8, 6))
        for clinoform_name in selected_plots:
            plot_data = plot_data_dict[clinoform_name]
            ax.plot(plot_data['x_data'], plot_data['y_data'], label=clinoform_name,
                    color=line_color, linestyle=line_style, marker=marker)

            # Plot upper and lower rollover points
            rollover_indices = plot_data['rollover_indices']
            if len(rollover_indices) >= 1:
                idx = rollover_indices[0]
                ax.plot(plot_data['x_data'][idx], plot_data['y_data'][idx], marker='*', color='gold', markersize=12, label=f"{clinoform_name} Upper Rollover")
            if len(rollover_indices) >= 2:
                idx = rollover_indices[1]
                ax.plot(plot_data['x_data'][idx], plot_data['y_data'][idx], marker='*', color='magenta', markersize=12, label=f"{clinoform_name} Lower Rollover")

            # Plot fitted models
            for model_name, model_data in plot_data['models'].items():
                ax.plot(model_data['x'], model_data['y'], label=f"{clinoform_name} - {model_name}", **model_data['style'])
        if show_legend:
            ax.legend(loc=legend_loc, fontsize=font_size)
        ax.set_xlabel('X', fontsize=font_size)
        ax.set_ylabel('Y', fontsize=font_size)
        ax.grid(True)
        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=plot_canvas_frame)
        toolbar = NavigationToolbar2Tk(canvas, plot_canvas_frame)
        toolbar.update()
        canvas.get_tk_widget().pack(fill='both', expand=True)
        canvas.draw()

        # Add mplcursors for interactive tooltips
        mplcursors.cursor(hover=True)

    elif plot_mode == 'Subplots':
        num_plots = len(selected_plots)
        fig, axes = plt.subplots(num_plots, 1, figsize=(8, 6 * num_plots))
        if num_plots == 1:
            axes = [axes]
        for ax, clinoform_name in zip(axes, selected_plots):
            plot_data = plot_data_dict[clinoform_name]
            ax.plot(plot_data['x_data'], plot_data['y_data'], label=clinoform_name,
                    color=line_color, linestyle=line_style, marker=marker)

            # Plot upper and lower rollover points
            rollover_indices = plot_data['rollover_indices']
            if len(rollover_indices) >= 1:
                idx = rollover_indices[0]
                ax.plot(plot_data['x_data'][idx], plot_data['y_data'][idx], marker='*', color='gold', markersize=12, label="Upper Rollover")
            if len(rollover_indices) >= 2:
                idx = rollover_indices[1]
                ax.plot(plot_data['x_data'][idx], plot_data['y_data'][idx], marker='*', color='magenta', markersize=12, label="Lower Rollover")

            # Plot fitted models
            for model_name, model_data in plot_data['models'].items():
                ax.plot(model_data['x'], model_data['y'], label=model_name, **model_data['style'])
            if show_legend:
                ax.legend(loc=legend_loc, fontsize=font_size)
            ax.set_xlabel('X', fontsize=font_size)
            ax.set_ylabel('Y', fontsize=font_size)
            ax.grid(True)
        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=plot_canvas_frame)
        toolbar = NavigationToolbar2Tk(canvas, plot_canvas_frame)
        toolbar.update()
        canvas.get_tk_widget().pack(fill='both', expand=True)
        canvas.draw()

        mplcursors.cursor(hover=True)

    elif plot_mode == 'Grid':
        num_plots = len(selected_plots)
        cols = 2
        rows = (num_plots + 1) // 2
        fig, axes = plt.subplots(rows, cols, figsize=(8 * cols, 6 * rows))
        axes = axes.flatten()
        for idx, clinoform_name in enumerate(selected_plots):
            ax = axes[idx]
            plot_data = plot_data_dict[clinoform_name]
            ax.plot(plot_data['x_data'], plot_data['y_data'], label=clinoform_name,
                    color=line_color, linestyle=line_style, marker=marker)

            # Plot upper and lower rollover points
            rollover_indices = plot_data['rollover_indices']
            if len(rollover_indices) >= 1:
                idx_point = rollover_indices[0]
                ax.plot(plot_data['x_data'][idx_point], plot_data['y_data'][idx_point], marker='*', color='gold', markersize=12, label="Upper Rollover")
            if len(rollover_indices) >= 2:
                idx_point = rollover_indices[1]
                ax.plot(plot_data['x_data'][idx_point], plot_data['y_data'][idx_point], marker='*', color='magenta', markersize=12, label="Lower Rollover")

            # Plot fitted models
            for model_name, model_data in plot_data['models'].items():
                ax.plot(model_data['x'], model_data['y'], label=model_name, **model_data['style'])
            if show_legend:
                ax.legend(loc=legend_loc, fontsize=font_size)
            ax.set_xlabel('X', fontsize=font_size)
            ax.set_ylabel('Y', fontsize=font_size)
            ax.grid(True)
        # Hide unused subplots
        for idx in range(len(selected_plots), len(axes)):
            fig.delaxes(axes[idx])
        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=plot_canvas_frame)
        toolbar = NavigationToolbar2Tk(canvas, plot_canvas_frame)
        toolbar.update()
        canvas.get_tk_widget().pack(fill='both', expand=True)
        canvas.draw()

        mplcursors.cursor(hover=True)

def preview():
    # Function to preview the selected data
    selected_clinoforms = [name for name, (x_col, y_col, rollover_col, var) in clinoform_vars.items() if var.get()]
    if not selected_clinoforms:
        messagebox.showerror("Error", "No clinoforms selected for preview.")
        return
    file_path = file_label.cget("text")
    sheet_name = sheet_name_combobox.get()
    if not file_path:
        messagebox.showerror("Error", "No file selected.")
        return

    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
            df.columns = df.columns.str.strip()
        else:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            df.columns = df.columns.str.strip()

        # Build a mapping of clinoform names to their columns
        clinoform_mapping = detect_clinoforms(df)
        clinoform_dict = {name: (x_col, y_col, rollover_col) for name, (x_col, y_col, rollover_col) in clinoform_mapping}

        # Show preview for the first selected clinoform
        clinoform_name = selected_clinoforms[0]
        x_col, y_col, rollover_col = clinoform_dict[clinoform_name]

        # Extract data
        x_data = df[x_col]
        y_data = df[y_col]

        # Create a new window to display the data
        preview_window = tk.Toplevel(root)
        preview_window.title(f"Data Preview - {clinoform_name}")

        # Create a Treeview to display the data
        columns = [x_col, y_col]
        if rollover_col:
            columns.append(rollover_col)
        tree_preview = ttk.Treeview(preview_window, columns=columns, show='headings')
        for col in columns:
            tree_preview.heading(col, text=col)
            tree_preview.column(col, width=100, anchor='center')

        # Insert data into the treeview
        for idx in df.index:
            values = [df.loc[idx, col] for col in columns]
            tree_preview.insert('', tk.END, values=values)

        tree_preview.pack(fill='both', expand=True)

        # Add a scrollbar
        scrollbar = ttk.Scrollbar(preview_window, orient='vertical', command=tree_preview.yview)
        tree_preview.configure(yscroll=scrollbar.set)
        scrollbar.pack(side='right', fill='y')

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred during data preview: {e}")
        logging.error(f"An error occurred during data preview: {e}")

def display_results_summary(df):
    # Clear existing rows
    for row in tree.get_children():
        tree.delete(row)
    # Insert new rows
    for _, row in df.iterrows():
        values = [row[col] for col in ['Clinoform'] + tree_columns[1:]]  # Adjusted to skip duplicate 'Clinoform'
        tree.insert('', tk.END, values=values)

def save_current_plot():
    filetypes = [('PNG Image', '*.png'), ('PDF Document', '*.pdf'), ('JPEG Image', '*.jpg'), ('Bitmap Image', '*.bmp'), ('SVG Image', '*.svg')]
    save_path = filedialog.asksaveasfilename(title="Save Plot", defaultextension='.png', filetypes=filetypes)
    if save_path:
        try:
            dpi = simpledialog.askinteger("DPI", "Enter DPI (dots per inch) for the output image:", minvalue=72, maxvalue=600, initialvalue=300)
            if dpi is None:
                dpi = 300  # default DPI
            fig = plt.gcf()
            fig.savefig(save_path, dpi=dpi)
            messagebox.showinfo("Save Plot", f"Plot saved successfully to {save_path}")
        except Exception as e:
            messagebox.showerror("Save Plot", f"Failed to save plot: {e}")

def create_plot_dashboard(parent):
    global plot_canvas_frame, plot_selection_listbox, plot_data_dict, plot_mode_var, show_legend_var
    global line_color_var, line_style_var, marker_var, legend_loc_var, font_size_var

    notebook_dashboard = ttk.Notebook(parent)
    notebook_dashboard.pack(fill="both", expand=True, padx=10, pady=10)

    plot_tab = ttk.Frame(notebook_dashboard)
    notebook_dashboard.add(plot_tab, text="Plot Visualization")
    plot_tab.rowconfigure(1, weight=1)
    plot_tab.columnconfigure(0, weight=1)

    plot_controls_frame = ttk.Frame(plot_tab)
    plot_controls_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=5)
    plot_controls_frame.columnconfigure(0, weight=1)

    ttk.Label(plot_controls_frame, text="Select Clinoform Plot(s):").pack(side="left", padx=5)

    plot_selection_listbox = tk.Listbox(plot_controls_frame, selectmode=tk.MULTIPLE, height=5)
    plot_selection_listbox.pack(side="left", padx=5, pady=5)
    plot_selection_listbox.bind('<<ListboxSelect>>', update_plot_display)

    plot_mode_var = tk.StringVar(value='Overlay')
    ttk.Label(plot_controls_frame, text="Plot Mode:").pack(side="left", padx=5)
    plot_modes = ['Overlay', 'Subplots', 'Grid']
    plot_mode_menu = ttk.OptionMenu(plot_controls_frame, plot_mode_var, plot_mode_var.get(), *plot_modes, command=lambda _: update_plot_display())
    plot_mode_menu.pack(side="left", padx=5)

    show_legend_var = tk.BooleanVar(value=True)
    show_legend_check = ttk.Checkbutton(plot_controls_frame, text="Show Legend", variable=show_legend_var, command=update_plot_display)
    show_legend_check.pack(side="left", padx=5)

    # Add options for line color and style
    ttk.Label(plot_controls_frame, text="Line Color:").pack(side="left", padx=5)
    line_color_var = tk.StringVar(value='blue')
    line_color_button = ttk.Button(plot_controls_frame, text="Choose Color", command=lambda: choose_color(line_color_var))
    line_color_button.pack(side="left", padx=5)

    ttk.Label(plot_controls_frame, text="Line Style:").pack(side="left", padx=5)
    line_styles = ['-', '--', '-.', ':']
    line_style_var = tk.StringVar(value='-')
    line_style_menu = ttk.OptionMenu(plot_controls_frame, line_style_var, line_style_var.get(), *line_styles, command=lambda _: update_plot_display())
    line_style_menu.pack(side="left", padx=5)

    ttk.Label(plot_controls_frame, text="Marker:").pack(side="left", padx=5)
    markers = ['o', 's', '^', 'D', '*', '.', 'None']
    marker_var = tk.StringVar(value='o')
    marker_menu = ttk.OptionMenu(plot_controls_frame, marker_var, marker_var.get(), *markers, command=lambda _: update_plot_display())
    marker_menu.pack(side="left", padx=5)

    # Legend location
    ttk.Label(plot_controls_frame, text="Legend Location:").pack(side="left", padx=5)
    legend_locations = ['best', 'upper right', 'upper left', 'lower left', 'lower right', 'right', 'center left',
                        'center right', 'lower center', 'upper center', 'center']
    legend_loc_var = tk.StringVar(value='best')
    legend_loc_menu = ttk.OptionMenu(plot_controls_frame, legend_loc_var, legend_loc_var.get(), *legend_locations, command=lambda _: update_plot_display())
    legend_loc_menu.pack(side="left", padx=5)

    # Font size
    ttk.Label(plot_controls_frame, text="Font Size:").pack(side="left", padx=5)
    font_size_var = tk.IntVar(value=10)
    font_size_spinbox = tk.Spinbox(plot_controls_frame, from_=6, to=20, textvariable=font_size_var, width=3, command=update_plot_display)
    font_size_spinbox.pack(side="left", padx=5)

    save_plot_button = ttk.Button(plot_controls_frame, text="Save Plot", command=save_current_plot)
    save_plot_button.pack(side="right", padx=5)

    plot_canvas_frame = ttk.Frame(plot_tab)
    plot_canvas_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=5)
    plot_canvas_frame.rowconfigure(0, weight=1)
    plot_canvas_frame.columnconfigure(0, weight=1)

# Create Main PanedWindow to divide left and right
main_paned = ttk.PanedWindow(root, orient=tk.HORIZONTAL)
main_paned.grid(row=0, column=0, sticky="nsew")

# Left PanedWindow for Controls and Results
left_pane = ttk.PanedWindow(main_paned, orient=tk.VERTICAL)
main_paned.add(left_pane, weight=1)

# Controls Frame
controls_frame = ttk.Frame(left_pane)
controls_frame.grid_rowconfigure(6, weight=1)  # Make the results frame expandable
controls_frame.columnconfigure(0, weight=1)
left_pane.add(controls_frame, weight=1)

# File selection controls
file_frame = ttk.Frame(controls_frame)
file_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=5)
file_frame.columnconfigure(1, weight=1)

browse_button = ttk.Button(file_frame, text="Browse for Excel or CSV File", command=open_file)
browse_button.grid(row=0, column=0, padx=5, pady=5)

file_label = ttk.Label(file_frame, text="No file selected")
file_label.grid(row=0, column=1, sticky="w")

# Sheet name selection
sheet_frame = ttk.Frame(controls_frame)
sheet_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=5)
sheet_frame.columnconfigure(1, weight=1)

ttk.Label(sheet_frame, text="Select Sheet (if Excel):").grid(row=0, column=0, padx=5, pady=5)
sheet_name_combobox = ttk.Combobox(sheet_frame, values=[], state="readonly")
sheet_name_combobox.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
sheet_name_combobox.bind("<<ComboboxSelected>>", on_sheet_selected)

# Clinoform selection
clinoform_frame = ttk.LabelFrame(controls_frame, text="Select Clinoforms")
clinoform_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=5)
clinoform_frame.columnconfigure(0, weight=1)

# Model selection
model_frame = ttk.LabelFrame(controls_frame, text="Select Fitting Models")
model_frame.grid(row=3, column=0, sticky="ew", padx=10, pady=5)
model_frame.columnconfigure(0, weight=1)

model_options = ['Linear', 'Quadratic', 'Exponential', 'Gaussian', 'Inverse Quadratic']
for idx, model_name in enumerate(model_options):
    var = tk.BooleanVar(value=False)
    chk = ttk.Checkbutton(model_frame, text=model_name, variable=var)
    chk.grid(row=idx, column=0, sticky="w", padx=5, pady=2)
    model_vars[model_name] = var

# Rollover handling options
rollover_frame = ttk.LabelFrame(controls_frame, text="Rollover Handling")
rollover_frame.grid(row=4, column=0, sticky="ew", padx=10, pady=5)
rollover_frame.columnconfigure(0, weight=1)

rollover_handling_var = tk.StringVar(value='auto')
ttk.Radiobutton(rollover_frame, text="Automatic Rollover Detection", variable=rollover_handling_var, value='auto').grid(row=0, column=0, sticky="w", padx=5, pady=2)
ttk.Radiobutton(rollover_frame, text="No Rollover", variable=rollover_handling_var, value='no_rollover').grid(row=1, column=0, sticky="w", padx=5, pady=2)

# Progress bar, Analyze, and Preview buttons
progress_frame = ttk.Frame(controls_frame)
progress_frame.grid(row=5, column=0, sticky="ew", padx=10, pady=5)
progress_frame.columnconfigure(0, weight=1)

progress = ttk.Progressbar(progress_frame, orient='horizontal', mode='determinate')
progress.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
progress_label = ttk.Label(progress_frame, text="Progress")
progress_label.grid(row=1, column=0, sticky="w", padx=5)

button_frame = ttk.Frame(progress_frame)
button_frame.grid(row=2, column=0, sticky="ew")
button_frame.columnconfigure(0, weight=1)

analyze_button = ttk.Button(button_frame, text="Analyze", command=analyze)
analyze_button.pack(side='right', padx=5, pady=5)

preview_button = ttk.Button(button_frame, text="Preview Data", command=preview)
preview_button.pack(side='left', padx=5, pady=5)

# Results display
results_frame = ttk.LabelFrame(controls_frame, text="Analysis Results")
results_frame.grid(row=6, column=0, sticky="nsew", padx=10, pady=5)
results_frame.columnconfigure(0, weight=1)
results_frame.rowconfigure(0, weight=1)

# Treeview for displaying results
tree_columns = ['Clinoform', 'Fitting Model', 'Equation', 'R²', 'RMSE', 'Confidence Intervals', 'Slope (Gradient)',
                'Topset Height', 'Topset Length', 'Foreset Height', 'Foreset Length',
                'Clinoform Height', 'Clinoform Width', 'Upper Rollover', 'Lower Rollover',
                'Concavity (Topset)', 'Concavity (Foreset)', 'Foreset Dip']

tree = ttk.Treeview(results_frame, columns=tree_columns, show='headings')
for col in tree_columns:
    tree.heading(col, text=col)
    tree.column(col, width=100, anchor='center')

tree.grid(row=0, column=0, sticky='nsew')

# Add a vertical scrollbar to the treeview
scrollbar = ttk.Scrollbar(results_frame, orient='vertical', command=tree.yview)
tree.configure(yscroll=scrollbar.set)
scrollbar.grid(row=0, column=1, sticky='ns')

# Optionally, add a horizontal scrollbar
h_scrollbar = ttk.Scrollbar(results_frame, orient='horizontal', command=tree.xview)
tree.configure(xscroll=h_scrollbar.set)
h_scrollbar.grid(row=1, column=0, sticky='ew')

# Menu bar with options
menu_bar = tk.Menu(root)
root.config(menu=menu_bar)

file_menu = tk.Menu(menu_bar, tearoff=0)
menu_bar.add_cascade(label="File", menu=file_menu)
file_menu.add_command(label="Open", command=open_file)
file_menu.add_command(label="Reset", command=reset_all)
file_menu.add_separator()
file_menu.add_command(label="Exit", command=root.quit)

help_menu = tk.Menu(menu_bar, tearoff=0)
menu_bar.add_cascade(label="Help", menu=help_menu)
help_menu.add_command(label="About")

# Right frame for plot dashboard
right_frame = ttk.Frame(main_paned)
right_frame.rowconfigure(0, weight=1)
right_frame.columnconfigure(0, weight=1)
main_paned.add(right_frame, weight=2)

# Create the plot dashboard
create_plot_dashboard(right_frame)

# Start the Tkinter event loop
root.mainloop()


