test-cnn:
	g++ test-cnn.cpp -I ../thirdparty/tiny-dnn/ -std=c++14 -lpthread `pkg-config --libs --cflags opencv` -lboost_system -lboost_filesystem -fopenmp -g -O2 -ltbb -mavx -o test-cnn