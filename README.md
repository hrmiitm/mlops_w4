# IRIS ML Pipeline with CI/CD

This project implements a complete MLOps pipeline for IRIS classification using:
- **DVC** for data and model versioning
- **GitHub Actions** for CI/CD automation
- **CML** for automated ML reports
- **pytest** for comprehensive testing
- **Google Cloud Storage** for remote storage

## Project Structure

iris-ml-pipeline/
├── .github/workflows/ # CI/CD pipelines
├── src/ # Source code
├── tests/ # Unit tests
├── models/ # Trained models (DVC tracked)
├── data/ # Data files (DVC tracked)
└── params.yaml # Model parameters