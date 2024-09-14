package test

import (
	"math/rand/v2"
	"testing"

	"github.com/ArtificialLegacy/go-webp/compression"
)

func TestCompression(t *testing.T) {
	for range 1000 {
		value := make([]byte, 100)
		prob := make([]uint8, 100)

		for i := range 100 {
			value[i] = byte(rand.Uint32N(256))
			prob[i] = uint8(rand.Uint32N(256))
		}

		encoder := compression.NewEncoder()
		for i := range 100 {
			encoder.Write(value[i], prob[i])
		}
		encoder.Flush()

		decoder := compression.NewDecoder(encoder.Output)
		for i := range 100 {
			v := decoder.ReadProb(8, prob[i])
			if v != uint32(value[i]) {
				t.Fatalf("Expected %v, got %v", value[i], v)
			}
		}
	}
}
