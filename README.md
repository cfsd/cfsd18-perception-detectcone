# cfsd18-perception-detectcone

[![Build Status](https://travis-ci.org/cfsd/cfsd18-perception-detectcone.svg?branch=master)](https://travis-ci.org/cfsd/cfsd18-perception-detectcone)

# Build and run image
## Build locally
'''
docker build -t detectcone:test -f Dockerfile.amd64 .
'''

## Run image
* There are three usecases, annotate, online and offline. When running the vehicle online, run the compose file in online folder, when running on the recording, run the compose file in the offline folder. When annotating data, run the compose file in annotate folder. 
'''
docker-compose up
'''

## After compose up
* This needs to be run every time after usecase finishs.
'''
docker-compose down
'''

# Annotate
* Run the compose file in usecases/annotate folder, an annotation folder will be created, together with 4 subfolders corresponding to 4 different classes. Go into each folder and check manually if all image belong to that class, if not, move the wrong ones to the correct folder.

* Generate data from the annotation folder. Run the cnn/generate_data.py, training data and validation data will be added to the data folder.
'''
python3 generate_data.py
'''

# Train
* Compile cnn/train.cpp to generate a runfile.
'''
g++ train.cpp -I ../thirdparty/tiny-dnn/ -std=c++14 -lpthread `pkg-config --libs --cflags opencv` -lboost_system -lboost_filesystem -fopenmp -g -O2 -ltbb -mavx -o train
'''
* Run runfile.
'''
./train data model 0.005 300 64
'''