# LegoSorterServer

Lego Sorter Server provides methods for detecting and classifying Lego bricks.

## How to run
1. Download the repository
```commandline
git clone https://github.com/LegoSorter/LegoSorterServer.git
```
2. Download network models for detecting lego bricks
```commandline
wget https://github.com/LegoSorter/LegoSorterServer/releases/download/1.0.0-alpha/detection_models.zip
wget https://github.com/LegoSorter/LegoSorterServer/releases/download/1.0.0-alpha/classification_models.zip
```

3. Extract models
```commandline
unzip detection_models.zip -d ./LegoSorterServer/lego_sorter_server/detection/models
unzip classification_models.zip -d ./LegoSorterServer/lego_sorter_server/classifier/models
```

4. Go to the root directory
```commandline
cd LegoSorterServer
```

5. Export *PYTHONPATH* environmental variable
```commandline
export PYTHONPATH=.
```

6. Run the server
```commandline
python lego_sorter_server
```

The server is now ready to handle requests. By default, the server listens on port *50051*

## Lego Sorter App
To test **Lego Sorter Server**, use the [Lego Sorter App](https://github.com/LegoSorter/LegoSorterApp), which is an application dedicated for this project.

## How to send a request (Optional)
**Lego Sorter Server** uses [gRPC](https://grpc.io/) to handle requests, the list of available methods is defined in `LegoSorterServer/lego_sorter_server/proto/LegoBrick.proto`.\
To call a method with your own client, look at [gRPC documentation](https://grpc.io/docs/languages/python/basics/#calling-service-methods)

