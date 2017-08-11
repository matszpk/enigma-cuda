## enigma-opencl

This is conversion of enigma-cuda written by Alex Shovkoplyas VE3NEA to OpenCL standard.
Currently this version is working with the AMD Radeon GPU's. Unfortunatelly, this version
have some severe problems on the NVIDIA GPU's.

Usage of this version is same like enigma-cuda. However, ability to manual choosing GPU
added to this version by setting CLDEV environment variable. To choose particular GPU device
just put number of this device counting from zero from first the OpenCL platform.
Ability to listing possible devices will be added later (just use clinfo to list it).

We recommend to run this program on Radeon GCN 1.0 and later GPU architecture
(all Radeon HD 7700/7800/7900, Radeon Rx 240-290, Radeon Rx 330-390/Fury/Nano,
Radeon Rx 400/VEGA and later).

## Assembly optimizations

This version introduces an assembly's optimization for GCN 1.0/1.1/1.2 architecture.
It requires the CLRadeonExtender libraries.

## Building

The scripts for 'GNU make' and 'NMake' are in 'linux' and 'windows' directory.
I did not add new executable to MS Visual Studio project. Just use 'make' or 'nmake' to
build enigma-opencl executable. Also, enigma-cuda executable for Linux can be built
using 'make' tool.
