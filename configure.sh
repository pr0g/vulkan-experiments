cmake -S . -B build/debug -G Ninja \
-DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
-DCMAKE_BUILD_TYPE=Debug \
-DCMAKE_PREFIX_PATH=$(pwd)/third-party/sdl/build \
-DAS_COL_MAJOR=ON -DAS_PRECISION_FLOAT=ON

cmake -S . -B build/release -G Ninja \
-DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
-DCMAKE_BUILD_TYPE=Release \
-DCMAKE_PREFIX_PATH=$(pwd)/third-party/sdl/build \
-DAS_COL_MAJOR=ON -DAS_PRECISION_FLOAT=ON
