# installation

## libtorch c++
https://docs.pytorch.org/cppdocs/installing.html

```sh
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=/absolute/path/to/libtorch ..
cmake --build . --config Release
```

run alphazero_training for training and alphazero for playing against the model

# Note
still doesn't work properly, work in progress
