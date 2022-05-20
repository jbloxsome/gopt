package main

import (
	"flag"
	"fmt"

	"github.com/jbloxsome/gopt"
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

	labels := []string{
		"false",
		"true",
	}

	model, err := gopt.NewGoPt(modelPath, labels)
	if err != nil {
		panic(err)
	}

	pred, err := gopt.Predict(imageFile)
	if err != nil {
		fmt.Println(err)
	}

	fmt.Println(pred)
}