echo "Compiling...";

g++ -O3 -march=native -mtune=intel -std=c++17 -mavx2 -mfma -flto -fopenmp -fPIC -fno-math-errno -I/usr/local/include/eigen3/ generateLookupTables.cpp -o generateLookupTables;

echo "Running...";

./generateLookupTables 0
./generateLookupTables 1
./generateLookupTables 2
./generateLookupTables 3
./generateLookupTables 4

./generateLookupTables 5
./generateLookupTables 6
./generateLookupTables 7
./generateLookupTables 8
./generateLookupTables 9

./generateLookupTables 10
./generateLookupTables 11
./generateLookupTables 12
./generateLookupTables 13
./generateLookupTables 14

./generateLookupTables 15
./generateLookupTables 16
./generateLookupTables 17
./generateLookupTables 18
./generateLookupTables 19