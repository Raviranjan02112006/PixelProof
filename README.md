PixelProof
AI-Generated Image Detection Using Deep Learning

PixelProof is a web-based image analysis project that combines
handcrafted mathematical image features, machine-learning classifiers, a
mathematical ensemble, and a ResNet50 deep-learning model to estimate
whether an image is real or AI-generated.
Research/demo disclaimer: PixelProof is an experimental detection
system. Its prediction is an estimate, not proof of an image's origin.
Results may be incorrect, especially for unfamiliar image sources,
editing, compression, or new generative models.

Features
- Upload an image through the web interface for analysis.
- Extract a set of mathematical and statistical image features.
- Run eight individual machine-learning models:
  - Logistic Regression
  - Random Forest
  - Support Vector Machine (SVM)
  - Gradient Boosting
  - MDMD-35
  - ZS-35
  - EDS-35
  - CFS-35
- Combine mathematical model evidence using a trained soft ensemble.
- Analyze the image using a ResNet50 CNN.
- Produce a hybrid prediction from mathematical-ensemble and CNN
  evidence.
- Display model-level results and a detection report.
Technology
- Python
- Flask
- TensorFlow / Keras
- scikit-learn
- NumPy and image-processing libraries
- HTML, CSS, and JavaScript
Exact package versions and additional dependencies are listed in
requirements.txt, if included in this repository.
Project structure
The repository may contain folders and files similar to the following:
PixelProof/
├── app.py
├── config.py
├── requirements.txt
├── README.md
├── models/
│   ├── individual/
│   ├── mathematical/
│   └── cnn/
├── src/
├── templates/
├── static/
├── uploads/
├── outputs/
└── reports/
The trained model files are required for inference. Keep the model
directory structure and filenames consistent with the paths configured
in the project.
Run locally
1. Clone the repository
git clone https://github.com/YOUR-USERNAME/PixelProof.git
cd PixelProof
Replace YOUR-USERNAME with the GitHub account that owns the
repository.
2. Create and activate a virtual environment
Windows PowerShell:
py -m venv .venv
.\.venv\Scripts\Activate.ps1
If PowerShell prevents activation, you can use the environment's Python
directly:
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Linux:
python3 -m venv .venv
source .venv/bin/activate
3. Install dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
If the repository does not contain requirements.txt, create one from
the working development environment before installation.
4. Start the application
python app.py
Open the local address shown in the terminal. If Flask is configured to
use port 5000, visit:
http://127.0.0.1:5000/
Keep the terminal open while using the application. Press Ctrl+C to
stop the server.
Model and inference notes
- PixelProof uses trained model artifacts stored under models/.
- The CNN model is based on ResNet50.
- The hybrid prediction combines mathematical and CNN evidence
  according to the project's configured weights.
- CPU inference is supported, although analysis may take longer than
  on a GPU.
- Do not change model filenames or folder paths unless the
  corresponding configuration and loading code are updated.
Privacy and safe use
- Do not commit .env files, credentials, private datasets, or
  personal images.
- Treat uploaded images and generated reports as potentially
  sensitive.
- Review .gitignore before pushing the repository.
- Do not expose Flask's development server directly to the public
  internet for production use. Use a production server and appropriate
  reverse-proxy/security configuration when deploying.
Limitations
- AI-image detection is an active research problem; no detector is
  perfectly reliable.
- A model's confidence score is not necessarily a calibrated
  probability.
- Performance depends on the training data, preprocessing, and image
  distribution.
- PixelProof should not be used as the sole basis for consequential
  decisions about authenticity.
Project status
Academic / research project under development.
License
No license has been specified yet. Unless a license is added to this
repository, reuse and redistribution are not automatically granted.