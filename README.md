# occlusion-and-interaction-aware-safe-control

Simulations for S. Gangadhar, Z. Wang, K. Poku, N. Yamada, K. Honda, Y. Nakahira, H. Okuda, T. Suzuki. An occlusion- and
interaction-aware safe control strategy for autonomous vehicles, submitted to The 22nd World Congress of the
International Federation of Automatic Control
 
 ## Structure

* the project root is a self-contained C++ code without a cmakelist
* no cmake or make required
* the code makes use of Eigen, a high-level C++ library of template headers for linear algebra
* an implementation of a QP-Solver is included as a header file


## Development

## Development
**Requirements:**
* Ubuntu 20.04
* GCC is required with atleast c++17
* Eigen 3.4 installed using cmake and make, see [Eigen: Getting Started][https://eigen.tuxfamily.org/dox/GettingStarted.html]
* Python3 with numpy, matplotlib, and scipy


## How to run

### To generate lookup tables
> . runAllLookupTables.sh

### To run simulations and generate results
> . runSimulations
