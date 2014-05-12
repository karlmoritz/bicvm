# Super simple Makefile to build  the project with cmake

.PHONY: all
all:
	mkdir -p build && \
	cd build && \
	cmake ../src && \
	make
