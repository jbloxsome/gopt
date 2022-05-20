package gopt

import (
	"log"
	"net/http"
	"errors"
	"os"

	"github.com/sugarme/gotch"
	"github.com/sugarme/gotch/ts"
	"github.com/sugarme/gotch/vision"
)

func GetFileContentType(path string) (string, error) {
	// Open File
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()

	// Only the first 512 bytes are used to sniff the content type.
	buffer := make([]byte, 512)

	_, err = f.Read(buffer)
	if err != nil {
		return "", err
	}

	// Use the net/http package's handy DectectContentType function. Always returns a valid
	// content-type by returning "application/octet-stream" if no others seemed to match.
	contentType := http.DetectContentType(buffer)

	return contentType, nil
}

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

func (gopt *GoPt) Predict(path string) (string, error) {
	// Check the file is an image first
	contentType, err := GetFileContentType(path)
	if err != nil {
		return "", err
	}

	if contentType != "image/jpeg" {
		return "", errors.New("must be an image file")
	}

	// Load the image file and resize it
	image, err := gopt.Imnet.LoadImageAndResize224(path)
	if err != nil {
		return "", err
	}

	// Apply the forward pass of the model to get the logits.
	unsqueezed := image.MustUnsqueeze(int64(0), false)
	image.MustDrop()
	raw_output := unsqueezed.ApplyCModule(gopt.Model)
	unsqueezed.MustDrop()
	output := raw_output.MustSoftmax(-1, gotch.Float, true)
	raw_output.MustDrop()

	// Convert to list of floats to represent label probabilities
	probs := output.Vals().([]float32)
	output.MustDrop()

	maxVal := probs[0]
	maxIndex := 0
	for i, v := range probs {
		if (v > maxVal) {
			maxVal = v
			maxIndex = i
		}
	}

	return gopt.Labels[maxIndex], nil
}