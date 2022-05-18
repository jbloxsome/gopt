package gopt

import (
	"log"

	"github.com/sugarme/gotch"
	"github.com/sugarme/gotch/ts"
	"github.com/sugarme/gotch/vision"
)

type GoPt struct {
	Model  *ts.CModule
	Labels []string
	Imnet  *vision.ImageNet
}

func (gopt *GoPt) LoadModel(path string) {
	// Create ImageNet object to use for input resizing
	gopt.Imnet = vision.NewImageNet()

	model, err := ts.ModuleLoadOnDevice(path, gotch.CPU)
	if err != nil {
		log.Fatal(err)
	}
	gopt.Model = model
}

func (gopt *GoPt) Predict(path string) string {
	// Load the image file and resize it
	image, err := gopt.Imnet.LoadImageAndResize224(path)
	if err != nil {
		log.Fatal(err)
	}

	// Apply the forward pass of the model to get the logits.
	output := image.MustUnsqueeze(int64(0), false).ApplyCModule(gopt.Model).MustSoftmax(-1, gotch.Float, true)

	// Convert to list of floats to represent label probabilities
	probs := output.Vals().([]float32)

	maxVal := probs[0]
	maxIndex := 0
	for i, v := range probs {
		if (v > maxVal) {
			maxVal = v
			maxIndex = i
		}
	}

	return gopt.Labels[maxIndex]
}