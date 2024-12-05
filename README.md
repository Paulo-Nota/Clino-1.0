# Clino 1.0

## Table of Contents

- [Overview](#overview)
- [Features](#features)
  - [Clinoform Analysis](#clinoform-analysis)
  - [Graph Data Extraction](#graph-data-extraction)
  - [Multi-Channel Profile Extraction](#multi-channel-profile-extraction)
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Clone the Repository](#clone-the-repository)
  - [Create a Virtual Environment (Recommended)](#create-a-virtual-environment-recommended)
  - [Install Dependencies](#install-dependencies)
- [Usage](#usage)
  - [Launching the Application](#launching-the-application)
  - [Clinoform Analysis Tool](#clinoform-analysis-tool)
  - [Graph Data Extraction](#graph-data-extraction)
  - [Multi-Channel Profile Extraction Tool](#multi-channel-profile-extraction-tool)
- [Data Requirements](#data-requirements)
  - [Clinoform Analysis](#clinoform-analysis-data-requirements)
  - [Graph Data Extraction](#graph-data-extraction-data-requirements)
  - [Profile Extraction](#profile-extraction-data-requirements)
- [Functionality](#functionality)
  - [Clinoform Analysis Functionality](#clinoform-analysis-functionality)
  - [Graph Data Extraction Functionality](#graph-data-extraction-functionality)
  - [Multi-Channel Profile Extraction Functionality](#multi-channel-profile-extraction-functionality)
- [Plot Visualization](#plot-visualization)
- [Logging](#logging)
- [Exporting Results](#exporting-results)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)
- [Contact](#contact)

## Overview

**Clino** is a comprehensive desktop application that integrates clinoform analysis, graph data extraction, and multi-channel profile extraction tools. Designed for geologists, researchers, and data analysts, Clino empowers users to efficiently import, analyze, visualize, and export geological and graphical data with ease. Leveraging Python's `tkinter` library for the user interface and powerful data processing libraries like `pandas`, `matplotlib`, and `scipy`, Clino offers a unified platform for robust data analysis and visualization.

## Features

### Clinoform Analysis

- **Easy Data Import**: Supports Excel (`.xlsx`, `.xls`) and CSV (`.csv`) file formats.
- **Dynamic Clinoform Detection**: Automatically identifies clinoforms based on column naming patterns.
- **Advanced Curve Fitting**: Fits multiple models including Linear, Quadratic, Exponential, Gaussian, and Inverse Quadratic.
- **Interactive Plotting**: Visualize data and fitted models with customizable plot options.
- **Rollover Handling**: Automatic detection or manual selection of rollover points.
- **Comprehensive Metrics**: Calculates R², RMSE, confidence intervals, slopes, concavity, and length, width, topset and foreset gradient.
- **Export Results**: Save analysis results to Excel files with organized sheets per clinoform.
- **Logging**: Maintains detailed logs of analysis processes and any errors encountered.

### Graph Data Extraction

- **Image Import**: Supports various image formats including JPEG, PNG, BMP, TIFF, and GIF.
- **Interactive Canvas**: Click on the image to add data points, set axes, and manage graphical elements.
- **Axis Setup**: Define X and Y axes by selecting points directly on the image.
- **Data Extraction**: Extract pixel-based data points and map them to graph coordinates based on defined axes.
- **Plot Visualization**: Generate and view plots of the extracted data within the application.
- **Customization**: Choose colors for data points and axis lines to enhance clarity and visual appeal.
- **Session Management**: Save and load sessions to preserve your work, including points, axes, and settings.
- **Undo/Redo Functionality**: Easily revert or reapply actions to manage your data extraction process.

### Multi-Channel Profile Extraction

- **Profile Drawing**: Draw various profiles such as linear, circular, semicircular, spline, freehand, rectangular, and elliptical on images.
- **Multi-Channel Support**: Extract profiles from channels like RGB, HSV, Grayscale, and Composite images.
- **Measurement Tools**: Measure lengths between points and areas of selected regions directly on the image.
- **Image Filtering**: Apply filters like Blur, Sharpen, and Edge Detection to enhance image quality.
- **ROI Selection**: Define Regions of Interest (ROI) for focused analysis.
- **Particle Analysis**: Threshold and analyze particles within the image, counting and outlining them.
- **Plot Visualization**: Generate plots of extracted profiles with detailed statistical annotations.
- **Session Management**: Save and load projects to preserve profiles, settings, and analyses.
- **Undo/Redo Functionality**: Easily revert or reapply actions to manage your workflow efficiently.
- **Export Data**: Save extracted profiles and measurement data to CSV files for further analysis.
- **Logging**: Maintain detailed logs of all actions and events for troubleshooting and record-keeping.

## Installation

### Prerequisites

- **Python 3.7 or higher**: Ensure Python is installed on your system. You can download it from [python.org](https://www.python.org/downloads/).

### Clone the Repository

```bash
git clone https://github.com/yourusername/clino.git
cd clino

