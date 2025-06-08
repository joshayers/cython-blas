#!/bin/bash

SRC_DIR="$1"

mkdir -p /project/blis
echo "/host$SRC_DIR"
cp /host$SRC_DIR /project/blis
cd /project/blis

export CONFIGURE_OPTS="$CONFIGURE_OPTS --disable-shared --enable-static"
export CONFIGURE_OPTS="$CONFIGURE_OPTS --disable-blas"
export CONFIGURE_OPTS="$CONFIGURE_OPTS --disable-cblas"
export CONFIGURE_OPTS="$CONFIGURE_OPTS --enable-threading=openmp"

./configure --prefix=/opt/blis $CONFIGURE_OPTS --enable-arg-max-hack x86_64
make -j2 V=1
make install

cp /opt/blis /host/opt/blis