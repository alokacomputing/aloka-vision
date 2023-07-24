# Aloka Vision

Aloka Vision is an innovative attendance system that utilizes realtime face recognition technology. It employs pretrained models from [InsightFace](https://github.com/deepinsight/insightface), a leading face analysis toolkit, to perform face detection and feature extraction from webcam or video file input. It also uses [Streamlit](https://streamlit.io/), a versatile app framework for data science and machine learning, to build a web-based user interface.

## Functionality

- Register new students and staff with their names and photos using the camera
- Recognize faces from live video or recorded footage
- Record attendance and generate reports with timestamps
- Explore InsightFace models and features in the playground folder with interactive notebooks

## Installation

To install the required dependencies, run:

```bash
pip install -r requirements.txt
```

## Usage

To run the web app, navigate to the app folder and run:

```bash
streamlit run Home.py
```

You can then access the app from your browser at http://localhost:8501

To run the playground notebooks, you need to install [Jupyter](https://jupyter.org/install) and [Google Colab](https://colab.research.google.com/notebooks/intro.ipynb). You can then open the notebooks from the playground folder and run them online or offline.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.