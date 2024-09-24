# Copyright Infringement Detection System

## Project Overview

The **Copyright Infringement Detection System** detects whether digital content violates copyright law by comparing it with existing copyrighted works. This system leverages **Machine Learning**, **Docker** for containerization, and **GitHub Actions** for continuous integration and deployment (CI/CD).



### Features
- Upload files for copyright infringement detection.
- Machine learning model detects similarities between the uploaded content and copyrighted material.
- Real-time analysis and report generation.
- API for submitting content and receiving results.
- Cloud storage and management on AWS S3.
- Fully containerized using Docker.
- CI/CD setup with GitHub Actions for automated testing and deployment.
- Technology Stack
- Backend: Django (v4.2.5)
- Database: MongoDB
- Cloud: AWS S3
- Containerization: Docker
- CI/CD: GitHub Actions
- Machine Learning: scikit-learn, evidently for monitoring
- Frontend: Basic HTML/CSS with Jinja2 template rendering.

## Project Structure

The project is organized as follows:

``` 

- **config/**: Directory containing project configuration files.
- **copyright_infringement/**: Django app with views, models, and routes for the application.
- **env/**: Folder for virtual environment setup.
- **.dockerignore**: Specifies files that Docker should ignore when building images.
- **.gitignore**: Specifies files that Git should ignore when tracking changes.
- **app.py**: The main script for running the application.
- **demo.py**: A demo file for testing and showcasing machine learning functionality.
- **Dockerfile**: Docker configuration file to containerize the app.
- **LICENSE**: Project licensing information.
- **README.md**: Documentation for the project.
- **requirements.txt**: Python dependencies for the project.
- **setup.py**: Script to package the project.
- **template.py**: A template file for setting up and extending functionalities.

```

## Overview of requirements

- **ipykernel**: Enables Jupyter Notebooks to run Python code in different environments.
- **pandas**: Data manipulation and analysis, mainly for structured datasets (e.g., CSV files).
- **numpy**: Efficient operations on arrays and matrices, crucial for numerical computations.
- **matplotlib**: Plotting library for creating static, animated, and interactive visualizations in Python.
- **plotly**: Interactive plotting library, useful for creating interactive graphs and dashboards.
- **seaborn**: Simplified interface for making attractive and informative statistical graphics (based on matplotlib).
- **scipy**: Collection of scientific and numerical algorithms (linear algebra, optimization, etc.).
- **scikit-learn**: Machine learning library for building models, handling data preprocessing, and feature extraction.
- **imblearn**: Provides techniques for handling imbalanced datasets, such as over/undersampling.
- **pymongo**: MongoDB client for connecting and interacting with MongoDB databases in Python.
- **boto3**: AWS SDK for Python, used for interacting with AWS services like S3, EC2, etc.
- **mypy-boto3-s3**: Type annotations for boto3, specifically for S3 operations, improving development accuracy.
- **botocore**: Core library underlying boto3, handling AWS service communication.
- **Django==4.2.5**: Web framework for building scalable, secure, and maintainable web applications.
- **djongo**: Integration tool to use MongoDB as a backend for Django applications.
- **from_root**: Utility for getting root paths in a project, useful for file handling.
- **dill**: Serialization library for saving and loading Python objects (extends Python’s pickle).
- **PyYAML**: Library for parsing and writing YAML, often used for configuration files.
- **jinja2**: Templating engine for Python, used in Django for rendering HTML templates dynamically.
- **python-multipart**: Handles file uploads in web applications, often used in form submissions.
- **-e .**: Installs the current project as a package in an editable form (development mode).
- **evidently==0.2.8**: Library for monitoring and evaluating machine learning model performance and data drift.

## Setup Instructions

### Prerequisites

Make sure you have the following installed:

- Python 3.9+
- MongoDB (local or cloud-based)
- AWS Account (for S3 storage)


## Step-by-Step Installation

1. Clone the repository:
``` bash 
git clone https://github.com/your-repo/copyright-infringement-detection.git
cd copyright-infringement-detection
```
2. Create and activate a virtual environment:
``` bash 
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```
3. Install the required dependencies:
``` bash
pip install -r requirements.txt
```
