package test

import (
	"image/png"
	"os"
	"testing"

	gowebp "github.com/ArtificialLegacy/go-webp"
)

const TEST_FILE = "test.webp"

func TestDecode(t *testing.T) {
	f, err := os.Open(TEST_FILE)
	if err != nil {
		t.Errorf("Error opening WebP file: %v", err)
	}
	defer f.Close()

	riffHeader, vp8Header, vp8FrameHeader, img, err := gowebp.Decode(f)
	if err != nil {
		t.Errorf("Error decoding WebP file: %v", err)
	}

	if riffHeader.Length != riffHeader.PaddedLength+12 {
		t.Errorf("Invalid length: expected %d, got %d", riffHeader.PaddedLength+12, riffHeader.Length)
	}

	if vp8Header.Version != gowebp.VP8_BICUBIC {
		t.Errorf("Invalid version: expected %d, got %d", gowebp.VP8_BICUBIC, vp8Header.Version)
	}

	if vp8Header.Width != 1050 {
		t.Errorf("Invalid width: expected %d, got %d", 1050, vp8Header.Width)
	}
	if vp8Header.Height != 700 {
		t.Errorf("Invalid height: expected %d, got %d", 700, vp8Header.Height)
	}

	if !vp8FrameHeader.UseSegment {
		t.Errorf("Invalid use_segment: expected %t, got %t", true, vp8FrameHeader.UseSegment)
	}

	if vp8FrameHeader.Partitions != 1 {
		t.Errorf("Invalid partitions: expected %d, got %d", 1, vp8FrameHeader.Partitions)
	}

	fpng, err := os.OpenFile("test.png", os.O_CREATE|os.O_TRUNC|os.O_RDWR, 0o666)
	if err != nil {
		t.Error(err)
	}
	defer fpng.Close()

	png.Encode(fpng, img)
}
