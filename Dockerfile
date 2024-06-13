# Step 1: Pull from miniconda
FROM continuumio/miniconda3

# Step 2: update and install necessary linux packages
RUN apt-get update && apt-get install -y \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Step 3: Copy env.yaml file to the container
COPY env.yaml /app/env.yaml

# Step 4: Create and activate conda environment with env.yaml
RUN conda env create -f /app/env.yaml

# Step 5: Activate the environment
RUN echo "source activate $(head -n 1 /app/env.yaml | cut -d ' ' -f2)" > ~/.bashrc
ENV PATH /opt/conda/envs/$(head -n 1 /app/env.yaml | cut -d ' ' -f2)/bin:$PATH

# Step 6: Copy src folder to the container
COPY . /app

RUN conda install pip

# RUN conda install mlflow
# Step 7: Get build for package
RUN pip install build mlflow

RUN python -m build /app

# Step 8: Install package
RUN pip install --default-timeout=100 dist/house_price_prediction-0.2.0-py3-none-any.whl

# Step 8: Start mlflow server
# CMD mlflow server --backend-store-uri mlruns/ --default-artifact-root mlruns/ --host 0.0.0.0 --port 5000 & \
#     python /app/src/scripts/main_script.py

EXPOSE 5000
RUN chmod +x entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]


# Building the image
# sudo docker build -t house_pred_image .

# Running the container
# sudo docker run -p 5000:5000 --name house_pred_container house_pred_image


# To push image to dockerhub

    # sudo docker login

    # sudo docker tag house_pred_image prithivee7/house_price_prediction:V3

    # sudo docker push prithivee7/house_price_prediction:V3
