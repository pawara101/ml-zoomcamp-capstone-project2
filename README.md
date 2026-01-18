# Air Quality Index (AQI) Prediction

## Problem Description

This project predicts the **Air Quality Index (AQI)** based on atmospheric pollutant levels using a deep learning model. 

### The Problem

Air quality is a critical environmental and health concern affecting millions worldwide. High levels of pollutants like CO, O3, NO2, SO2, PM10, and PM2.5 can cause respiratory diseases and other health issues. This project aims to build a machine learning model that can:

- Predict AQI scores from pollutant concentration measurements
- Help cities and environmental agencies forecast air quality
- Enable early warnings for poor air quality conditions

### Dataset

The model is trained on the **Air Quality Data (2019-2025)** dataset from Kaggle, which contains:
- Historical air quality measurements from 51 US cities
- Pollutant levels: CO, O3, NO2, SO2, PM10, PM2.5
- Target variable: AQI (Air Quality Index)

### Model Architecture

The project uses a **Convolutional Neural Network (CNN)** implemented in PyTorch:
- 1D Convolution layers for feature extraction
- ReLU activation functions
- Fully connected layers for prediction
- Optimized for regression to predict continuous AQI values

## How to Run the Project

### Prerequisites

- Python 3.8 or higher
- pip or conda package manager
- Docker (optional, for containerized deployment)

### Local Setup

#### 1. Clone or Download the Project

```bash
cd ml-zoomcamp-capstone-project2
```

#### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

#### 3. Train the Model (Optional)

If you want to retrain the model from scratch:

```bash
python train.py
```

This will:
- Load the Air Quality dataset from `data/Air_Quality_Data.csv`
- Preprocess the data (encode cities, extract date features)
- Split into train/test sets (80/20)
- Train the CNN model
- Save the trained model to `models/model.pth`

#### 4. Run the Prediction Web App

Start the Streamlit web interface:

```bash
streamlit run predict.py
```

Then open your browser to `http://localhost:8501` and:
- Select a city
- Input pollutant levels (CO, O3, NO2, SO2, PM10, PM2.5)
- Click predict to get the AQI score

### Docker Deployment

#### Build the Docker Image

```bash
docker build -t aqi-predictor .
```

#### Run the Container

```bash
docker run -p 8501:8501 aqi-predictor
```

Access the app at `http://localhost:8501`

## Project Structure

```
├── notebook.ipynb          # Jupyter notebook with model development
├── train.py               # Training script
├── predict.py             # Streamlit prediction app
├── requirements.txt       # Python dependencies
├── Dockerfile             # Docker configuration
├── models/
│   └── model.pth         # Trained PyTorch model
└── data/
    └── Air_Quality_Data.csv  # Training dataset
```

## Files Description

- **train.py**: Trains the CNN model on the air quality dataset
- **predict.py**: Streamlit web application for making predictions
- **notebook.ipynb**: Interactive exploration and model development
- **model.pth**: Pre-trained model weights

## Performance

The model is trained to predict AQI scores with optimal accuracy using the pollutant measurements and city information as features.

## Technologies Used

- **PyTorch**: Deep learning framework
- **Pandas & Scikit-learn**: Data preprocessing
- **Streamlit**: Web interface for predictions
- **Docker**: Container deployment

## License

This project is part of the ML ZoomCamp Capstone Project 2.