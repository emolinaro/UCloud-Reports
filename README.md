# UCloud Services: Analysis Reports

This repository collects analysis reports on the usage of UCloud storage and computing services. 

Each report is built as an interactive `streamlit` web app, which can be deployed locally via Docker.

To build the Docker image, run the script:
```bash
./Docker/docker-build
```

## 2020 Reports

### 1. MLPerf Benchmark tests

Deploy the report locally with the command:
```bash
cd 2020
docker run --rm  -v $PWD:/work -p 8501:8501 streamlit:latest start_webapp -F /work/MLPerf-Benchmarks -f /work/MLPerf-Benchmarks/main.py
```
This report can be accessed at `http://localhost:8501`.
