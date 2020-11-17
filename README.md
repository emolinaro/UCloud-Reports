# UCloud Services: Analysis Reports

This repository collects comparative performance analyses between UCloud computing services and external systems. 

Each report is built as an interactive `streamlit` web app, which can be deployed locally in a Docker container.
To build the image, run the script:
```bash
./Docker/docker-build
```

## 2020 Reports

### 1.

Deploy the report locally with the command:
```bash
cd 2020
docker run --rm  -v $PWD:/work -p 8501:8501 streamlit:latest start_webapp -F /work/1/ -f /work/1/main.py
```
This report can be accessed at: `http://localhost:8501`.

### 2.

Deploy the report locally with the command:
```bash
cd 2020
docker run --rm  -v $PWD:/work -p 8501:8501 streamlit:latest start_webapp -F /work/2/ -f /work/2/main.py
```
This report can be accessed at: `http://localhost:8501`.

### 3.

Deploy the report locally with the command:
```bash
cd 2020
docker run --rm  -v $PWD:/work -p 8501:8501 streamlit:latest start_webapp -F /work/3/ -f /work/3/main.py
```
This report can be accessed at: `http://localhost:8501`.
