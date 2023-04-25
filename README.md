# LegoSorterServer

Lego Sorter Server provides methods for detecting and classifying Lego bricks.

This branch require LegoSorterWeb to be started before LegoSorterServer.

## PL Native run Ubuntu + Linux
https://gist.github.com/uiopak/691841ea5c2e1d6d57c8f959010a1b35

## How to run
1. Download the repository
```commandline
git clone https://github.com/LegoSorter/LegoSorterServer.git
```
2. Download network models for detecting lego bricks
```commandline
wget https://github.com/LegoSorter/LegoSorterServer/releases/download/1.2.0/detection_models.zip
wget https://github.com/LegoSorter/LegoSorterServer/releases/download/1.2.0/classification_models.zip
```

3. Extract models
```commandline
unzip detection_models.zip -d ./LegoSorterServer/lego_sorter_server/analysis/detection/models
unzip classification_models.zip -d ./LegoSorterServer/lego_sorter_server/analysis/classification/models
```
4. Add new test models

`yolov5_n.pt` nano copy to `lego_sorter_server/analysis/detection/models/yolo_model`:

https://drive.google.com/file/d/1e-u_CquSOqpXpsqEbzc-wLq87NNyCBbr/view?usp=sharing

`tinyvit.pth` copy to `lego_sorter_server/analysis/classification/models/tiny_vit_model`: 

https://drive.google.com/file/d/19P-_-lbf0Chj8Bm4MxLW-jxGwr4EYnSi/view?usp=share_link

YOLOv5 detection model can be changed in: 
`lego_sorter_server/analysis/detection/detectors/YoloLegoDetectorFast.py`
`lego_sorter_server/analysis/detection/detectors/YoloLegoDetector.py`

Classification model is changed by setting *LEGO_SORTER_CLASSIFIER* environmental variable:
* LEGO_SORTER_CLASSIFIER=0 - use old EfficientNet model running using Keras
* LEGO_SORTER_CLASSIFIER=1 - use new test TinyViT model running using PyTorch (Default when variable isn't set)

Example:

Linux:
```bash
export LEGO_SORTER_CLASSIFIER=1
```

Windows (PowerShell):
```powershell
$env:LEGO_SORTER_CLASSIFIER=1
```

or
changing default value in `lego_sorter_server/__main__.py`
 
5. Go to the root directory
```commandline
cd LegoSorterServer
```

6. Create conda environment

Nvidia GPU with CUDA (default name `lego`):
```commandline
conda env create --file environment.yml
```
or
```commandline
conda env create -f environment.yml
```
For CPU only use `environment-cpu.yml` (smaller environment)

7. Activate conda environment
```powershell
conda env create -f environment-cpu.yml
```

8. Export *PYTHONPATH* environmental variable

Linux:
```bash
export PYTHONPATH=.
```

Windows (PowerShell):
```powershell
$env:PYTHONPATH="."
```

9. Run the server
```commandline
python lego_sorter_server
```
If LegoSorterWeb is running on different machine/port set *LEGO_SORTER_WEB_ADDRESS* environmental variable

LegoSorterWeb when run after publish will run on port 5000 default, but on 5002 if run in VS or using `dotnet run`

Linux:
```bash
export LEGO_SORTER_WEB_ADDRESS='http://192.168.11.189:5000'
```

Windows (PowerShell):
```powershell
$env:LEGO_SORTER_WEB_ADDRESS='http://192.168.11.189:5000'
```

The server is now ready to handle requests. By default, the server listens on port *50051*, *50052*, *5151* and *5005*

## How to run Docker (CPU only)
1. Follow 1 - 5 [How to run](#how-to-run)
2. Build image
```commandline
docker build -t lego -f Dockerfile .
```
3. Create two folders for persistent data

4. Change folders permission to `777` (linux only)

5. Create container

Linux example, change folder paths `/home/ubuntu/lego/images` and `/home/ubuntu/lego/sqlite_file` and set IP off computer running LegoSorterWeb `--env LEGO_SORTER_WEB_ADDRESS=http://192.168.11.189:5002`:
```commandline
docker run -d --name lego -p 50051:50051 -p 50052:50052 -p 5151:5151 -p 5005:5005 \
--restart unless-stopped --env LOG_LEVEL=DEBUG --env LEGO_SORTER_WEB_ADDRESS='http://192.168.11.189:5002' \
--mount type=bind,source=/home/ubuntu/lego/images,destination=/app/lego_sorter_server/images/storage/stored \
--mount type=bind,source=/home/ubuntu/lego/sqlite_file,destination=/app/lego_sorter_server/database/sqlite_file \
lego
```
Windows (PowerShell) example, change folder paths `//g/LEGO/docker/images` and `//g/LEGO/docker/sqlite_file` and set IP off computer running LegoSorterWeb `--env LEGO_SORTER_WEB_ADDRESS='http://192.168.11.189:5002'`:
```powershell
docker run -d --name lego -p 50051:50051 -p 50052:50052 -p 5151:5151 -p 5005:5005 `
--restart unless-stopped --env LOG_LEVEL=DEBUG --env LEGO_SORTER_WEB_ADDRESS='http://192.168.11.189:5002' `
--mount type=bind,source=//g/LEGO/docker2/images,destination=/app/lego_sorter_server/images/storage/stored `
--mount type=bind,source=//g/LEGO/docker2/sqlite_file,destination=/app/lego_sorter_server/database/sqlite_file `
lego
```
## Lego Sorter App
To test **Lego Sorter Server**, use the [Lego Sorter App](https://github.com/LegoSorter/LegoSorterApp), which is an application dedicated for this project.

## How to send a request (Optional)
**Lego Sorter Server** uses [gRPC](https://grpc.io/) to handle requests, the list of available methods is defined in `LegoSorterServer/lego_sorter_server/proto/LegoBrick.proto`.\
To call a method with your own client, look at [gRPC documentation](https://grpc.io/docs/languages/python/basics/#calling-service-methods)

