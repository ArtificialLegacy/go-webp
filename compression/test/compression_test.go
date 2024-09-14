package test

import (
	"math/rand/v2"
	"testing"

	"github.com/ArtificialLegacy/go-webp/compression"
)

func TestCompression(t *testing.T) {
	for range 1000 {
		value := byte(rand.Uint32())
		prob := uint8(rand.Uint32())

		encoder := compression.NewEncoder()
		for range 100 {
			encoder.Write(value, prob)
		}
		encoder.Flush()

		decoder := compression.NewDecoder(encoder.Output)
		for range 100 {
			v := decoder.Read(8, prob)
			if v != uint32(value) {
				t.Fatalf("Expected %v, got %v", value, v)
			}
		}
	}
}
