package main

import (
	"flag"
	"log"
	"fmt"

	"github.com/sugarme/gotch"
	"github.com/sugarme/gotch/ts"
	"github.com/sugarme/gotch/vision"
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

	imageNet := vision.NewImageNet()

	// Load the image file and resize it
	image, err := imageNet.LoadImageAndResize224(imageFile)
	if err != nil {
		log.Fatal(err)
	}

	// Load the Python saved module.
	model, err := ts.ModuleLoadOnDevice(modelPath, gotch.CPU)
	if err != nil {
		log.Fatal(err)
	}

	// Apply the forward pass of the model to get the logits.
	output := image.MustUnsqueeze(int64(0), false).ApplyCModule(model).MustSoftmax(-1, gotch.Float, true)

	// top := imageNet.Top(output, int64(2))

	fmt.Println(output.Vals().([]float32)[0])
}