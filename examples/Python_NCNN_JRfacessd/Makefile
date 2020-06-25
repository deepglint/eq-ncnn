BOOST_INCLUDE_PATH=/usr/include
BOOST_LIB_PATH=/usr/lib
NCNN_INCLUDE_PATH=../../build-linux/install/include
NCNN_LIB_PATH=../../build-linux/install/lib
PYTHON_INCLUDE_PATH=/root/anaconda3/include/python3.6m
PYTHON_LIB_PATH=/root/anaconda3/lib 

all:
	g++ -fPIC -std=c++11  -fpermissive -I ${BOOST_INCLUDE_PATH} -I ${PYTHON_INCLUDE_PATH} -I ${NCNN_INCLUDE_PATH} -c ncnn.cpp 
	g++ -fPIC -std=c++11 -fpermissive -I ${BOOST_INCLUDE_PATH} -I ${PYTHON_INCLUDE_PATH} -I ${NCNN_INCLUDE_PATH} -c ncnn2python.cpp 
	g++ -shared -std=c++11  -o ncnn.so  ncnn.o ncnn2python.o -L ${BOOST_LIB_PATH} -lboost_python-py35 -L ${PYTHON_LIB_PATH} -lpython3.6m  -L ${NCNN_LIB_PATH} -lncnn
	cp ncnn.so /root/anaconda3/lib/python3.6/site-packages/
	

clean:
	rm ncnn.so  ncnn.o ncnn2python.o
