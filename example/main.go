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