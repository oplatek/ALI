#!/bin/bash

DIR=stability_out
mkdir -p $DIR

for s in 0.001 0.005 0.01 0.05 0.1 0.5 1.0 5.0 10.0 50.0 100.0; do
	echo 'Running sigma=' $s
	THEANORC=theanorc ./stability.py cifar10 ali_cifar10.tar --save-path $DIR/sigma_$s.png --sigma $s > /dev/null 2>&1
done

