FROM mcr.microsoft.com/azure-functions/python:4-python3.12

ENV AzureWebJobsScriptRoot=/home/site/wwwroot \
    AzureFunctionsJobHost__Logging__Console__IsEnabled=true

RUN apt-get update \
    && apt-get install -y --no-install-recommends g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

COPY . /home/site/wwwroot
WORKDIR /home/site/wwwroot

RUN suffix="$(python -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")" \
    && includes="$(python -m pybind11 --includes)" \
    && g++ -O3 -Wall -shared -std=c++17 -fPIC -pthread $includes \
        native/processor.cpp -o "tools/native_engine${suffix}" \
    && g++ -O3 -Wall -shared -std=c++17 -fPIC -pthread $includes \
        native/tensor_engine.cpp -o "tools/tensor_engine${suffix}"
