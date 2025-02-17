# Copyright Infringement Detection System

## Project Overview

The **Copyright Infringement Detection System** detects whether digital content violates copyright law by comparing it with existing copyrighted works. This system leverages **Machine Learning**, **Docker** for containerization, and **GitHub Actions** for continuous integration and deployment (CI/CD).



### Features
- Upload files for copyright infringement detection.
- Machine learning model detects similarities between the uploaded content and copyrighted material.
- Real-time analysis and report generation.
- API for submitting content and receiving results.
- Fully containerized using Docker.
- CI/CD setup with GitHub Actions for automated testing and deployment.
### Technology Stack
- Backend: Flask 
- Database: MongoDB
- Containerization: Docker
- CI/CD: GitHub Actions
- Machine Learning: scikit-learn, evidently for monitoring
- Frontend: Basic HTML/CSS with Jinja2 template rendering.

## Project Structure

The project is organized as follows:

 
``` bash
├── config/                      # Configuration files for the project
├── copyright_infringement/      # Django app folder containing views, models, and URLs
├── env/                         # Virtual environment folder (recommend placing outside project directory)
├── app.py                       # Main entry point for the application
├── demo.py                      # Demo script for testing machine learning models
├── Dockerfile                   # Docker configuration for containerizing the application
├── LICENSE                      # License for the project
├── README.md                    # Project documentation
├── requirements.txt             # Python package dependencies
├── setup.py                     # Setup file for packaging
├── template.py                  # Base script used for setting up the project
```

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
## Project Workflow
1. constants
2. entity
3. components
4. pipeline
## Set up your Mongodb atlas cloud and add your credintials
``` bash
$Env:MONGODB_USERNAME = "YOURUSERNAME"
$ENV:MONDB_PASSWORD = "YOURPASSWORD"
```
## Run the pipeline
``` bash
python test.py
```
```
python copyright_infringement\components\image_data_ingestion.py
```
## Start the flask 
```bash 
python app.py
```
## To build docker image 
```
docker-compose build
```
## To run the docker 
```
docker run -p 8000:8000 <your image name>
```



