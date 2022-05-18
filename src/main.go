package main

import (
	"fmt"

	"github.com/sugarme/gotch"
	"github.com/sugarme/gotch/ts"
)

func main() {
	xs := ts.MustRand([]int64{3, 5, 6}, gotch.Float, gotch.CPU)
	fmt.Printf("%8.3f\n", xs)
	fmt.Printf("%i", xs)
}