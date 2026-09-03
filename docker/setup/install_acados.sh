#!/bin/bash

echo "Installing acados from source"

echo "Cloning acados repository"
cd /
git clone https://github.com/acados/acados.git
cd acados

git submodule update --recursive --init --depth=1

mkdir -p build
cd build

echo "Building acados with cmake"
cmake -DACADOS_WITH_QPOASES=ON  \
      -DACADOS_WITH_HPIPM=ON \
      -DACADOS_WITH_OSQP=ON .. && make install

cd /

echo "Installing acados Python interface"
pip install /acados/interfaces/acados_template

echo "Finished installing acados"
