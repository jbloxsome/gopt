package gopt

import (
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

func NewGoPt(modelPath string, labels []string) (*GoPt, error) {
	model, err := ts.ModuleLoadOnDevice(modelPath, gotch.CPU)
	if err != nil {
		return nil, err
	}

	imageNet := vision.NewImageNet()

	return &GoPt{
		Model:  model,
		Labels: labels,
		Imnet: imageNet,
	}, nil
}

func (gopt *GoPt) Predict(path string) (string, error) {
	// Check the file is an image first
	contentType, err := GetFileContentType(path)
	if err != nil {
		return "", err
	}

	if contentType != "image/jpeg" && contentType != "image/png" && contentType != "image/gif" {
		return "", errors.New("must be an image file")
	}

	// Load the image file and resize it
	image, err := gopt.Imnet.LoadImageAndResize224(path)
	if err != nil {
		return "", err
	}

	// Apply the forward pass of the model to get the logits.
	input := image.MustUnsqueeze(0, false)
	raw_output := input.ApplyCModule(gopt.Model)
	output := raw_output.MustSoftmax(-1, gotch.Float, true)

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