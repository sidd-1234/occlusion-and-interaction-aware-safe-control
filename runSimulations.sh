python3 decimateAndFilter.py

echo "Compiling...";

g++ -O3 -march=native -mtune=intel -std=c++17 -mavx2 -mfma -flto -fopenmp -fPIC -fno-math-errno -I/usr/local/include/eigen3/ safeOcclusionControl.cpp -o safeOcclusionControl;

echo "Running...";

./safeOcclusionControl 0
./safeOcclusionControl 1
./safeOcclusionControl 2
./safeOcclusionControl 3
./safeOcclusionControl 4

./safeOcclusionControl 5
./safeOcclusionControl 6
./safeOcclusionControl 7
./safeOcclusionControl 8
./safeOcclusionControl 9

./safeOcclusionControl 10
./safeOcclusionControl 11
./safeOcclusionControl 12
./safeOcclusionControl 13
./safeOcclusionControl 14

./safeOcclusionControl 15
./safeOcclusionControl 16
./safeOcclusionControl 17
./safeOcclusionControl 18
./safeOcclusionControl 19

echo "Plotting...";

python3 plotResults.py