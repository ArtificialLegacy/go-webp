package gowebp

import (
	"encoding/binary"
	"fmt"
	"io"
)

const headerSize = 20

// ASCII headers are in big-endian format
const (
	RIFF_HEADER = 0x52_49_46_46 // RIFF
	WEPB_HEADER = 0x57_45_42_50 // WEBP
	VP8_HEADER  = 0x56_50_38_20 // VP8(space)
	VP8L_HEADER = 0x56_50_38_4C // VP8L
)

type FormatError string

func (e FormatError) Error() string { return "webp: invalid format: " + string(e) }

func NewFormatError(err string) FormatError {
	return FormatError(err)
}

func NewFormatErrorf(format string, args ...interface{}) FormatError {
	return FormatError(fmt.Sprintf(format, args...))
}

type RIFFHeader struct {
	Length       uint32 // + 12
	PaddedLength uint32
}

func Decode(r io.Reader) (*RIFFHeader, *VP8Header, error) {
	riffHeader, err := decodeRiffHeader(r)
	if err != nil {
		return nil, nil, err
	}

	vp8Header, err := decodeVP8FrameHeader(r)
	if err != nil {
		return nil, nil, err
	}

	return riffHeader, vp8Header, nil
}

func decodeRiffHeader(r io.Reader) (*RIFFHeader, error) {
	header := make([]byte, 20)
	n, err := r.Read(header)
	if err != nil {
		return nil, NewFormatErrorf("error reading header: %v", err)
	}
	if n != headerSize {
		return nil, NewFormatErrorf("invalid header: expected %d bytes, got %d", headerSize, n)
	}

	riffHeader := binary.BigEndian.Uint32(header[0:4])
	if riffHeader != RIFF_HEADER {
		return nil, NewFormatErrorf("invalid RIFF header, expected %d, got %d", RIFF_HEADER, riffHeader)
	}

	length := binary.LittleEndian.Uint32(header[4:8])

	webpHeader := binary.BigEndian.Uint32(header[8:12])
	if webpHeader != WEPB_HEADER {
		return nil, NewFormatError("invalid WEBP header")
	}

	vp8Header := binary.BigEndian.Uint32(header[12:16])
	if vp8Header != VP8_HEADER {
		if vp8Header == VP8L_HEADER {
			return nil, NewFormatError("VP8L not supported")
		}

		return nil, NewFormatError("invalid VP8 header")
	}

	paddedLength := binary.LittleEndian.Uint32(header[16:20])

	return &RIFFHeader{
		Length:       length,
		PaddedLength: paddedLength,
	}, nil
}

const START_CODE = 0x9d012a

type VP8Header struct {
	Version  uint8
	PartSize uint32

	Width  uint16
	Height uint16
}

const (
	VP8_BICUBIC         = 0
	VP8_BILINEAR_SIMPLE = 1
	VP8_BILINEAR        = 2
	VP8_NONE            = 3
)

func decodeVP8FrameHeader(r io.Reader) (*VP8Header, error) {
	header := make([]byte, 12)
	n, err := r.Read(header[0:3])
	if err != nil {
		return nil, NewFormatErrorf("error reading VP8 frame header: %v", err)
	}
	if n != 3 {
		return nil, NewFormatErrorf("invalid VP8 frame header: expected %d bytes, got %d", 3, n)
	}

	data := binary.LittleEndian.Uint32(header[0:4])

	keyFrame := data & 0x1
	version := (data >> 1) & 0x7
	showFrame := (data >> 4) & 0x1
	partSize := (data >> 5) & 0x7FFFF

	if keyFrame != 0 {
		return nil, NewFormatError("invalid VP8 frame header: must be a key frame")
	}
	if showFrame != 1 {
		return nil, NewFormatError("invalid VP8 frame header: frame must be shown")
	}

	n, err = r.Read(header[5:8])
	if err != nil {
		return nil, NewFormatErrorf("error reading VP8 frame header: %v", err)
	}
	if n != 3 {
		return nil, NewFormatErrorf("invalid VP8 frame header: expected %d bytes, got %d", 3, n)
	}

	startCode := binary.BigEndian.Uint32(header[4:8])
	if startCode != START_CODE {
		return nil, NewFormatErrorf("invalid start code: expected %d, got %d", START_CODE, startCode)
	}

	n, err = r.Read(header[8:12])
	if err != nil {
		return nil, NewFormatErrorf("error reading VP8 frame header: %v", err)
	}
	if n != 4 {
		return nil, NewFormatErrorf("invalid VP8 frame header: expected %d bytes, got %d", 4, n)
	}

	width := binary.LittleEndian.Uint16(header[8:10])
	height := binary.LittleEndian.Uint16(header[10:12])

	h := &VP8Header{
		Version:  uint8(version),
		PartSize: partSize,
		Width:    width,
		Height:   height,
	}

	return h, nil
}
