# GoPT

## Overview
GoPt is a small library for loading PyTorch image classification models into Golang code.

## Dependencies

- **Libtorch** C++ v1.11.0 library of [Pytorch](https://pytorch.org/)

## Installation

- Default CUDA version is `11.3` if CUDA is available otherwise using CPU version.
- Default Pytorch C++ API version is `1.11.0`

**NOTE**: `libtorch` will be installed at **`/usr/local/lib`**

### CPU

#### Step 1: Setup libtorch (skip this step if a valid libtorch already installed in your machine!)

```bash
    wget https://raw.githubusercontent.com/sugarme/gotch/master/setup-libtorch.sh
    chmod +x setup-libtorch.sh
    export CUDA_VER=cpu && bash setup-libtorch.sh
```

**Update Environment**: in Debian/Ubuntu, add/update the following lines to `.bashrc` file

```bash
    export GOTCH_LIBTORCH="/usr/local/lib/libtorch"
    export LIBRARY_PATH="$LIBRARY_PATH:$GOTCH_LIBTORCH/lib"
    export CPATH="$CPATH:$GOTCH_LIBTORCH/lib:$GOTCH_LIBTORCH/include:$GOTCH_LIBTORCH/include/torch/csrc/api/include"
    export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$GOTCH_LIBTORCH/lib"
```

#### Step 2: Setup gotch

```bash
    wget https://raw.githubusercontent.com/sugarme/gotch/master/setup-gotch.sh
    chmod +x setup-gotch.sh
    export CUDA_VER=cpu && export GOTCH_VER=v0.7.0 && bash setup-gotch.sh
```

## Example
```
package main

import (
	"flag"
	"fmt"

	"github.com/jbloxsome/gopt/gopt"
)

var (
	modelPath string
	imageFile string
)

func init() {
	flag.StringVar(&modelPath, "modelpath", "./model.pt", "full path to exported pytorch model.")
	flag.StringVar(&imageFile, "image", "./image.jpg", "full path to image file.")
}

func main() {
	flag.Parse()

	gopt := gopt.GoPt{
		Labels: []string{
			"false",
			"true",
		},
	}

	gopt.LoadModel(modelPath)

	pred := gopt.Predict(imageFile)

	fmt.Println(pred)
}
```