package gowebp

import (
	"encoding/binary"
	"fmt"
	"io"

	"github.com/ArtificialLegacy/go-webp/compression"
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

func Decode(r io.Reader) (*RIFFHeader, *VP8Header, *VP8FrameHeader, error) {
	riffHeader, err := decodeRiffHeader(r)
	if err != nil {
		return nil, nil, nil, err
	}

	vp8Header, err := decodeVP8Header(r)
	if err != nil {
		return nil, nil, nil, err
	}

	vp8FrameHeader, err := decodeVP8Frame(r, int(vp8Header.PartSize))
	if err != nil {
		return nil, nil, nil, err
	}

	return riffHeader, vp8Header, vp8FrameHeader, nil
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
	XScale uint16
	Height uint16
	YScale uint16
}

const (
	VP8_BICUBIC         = 0
	VP8_BILINEAR_SIMPLE = 1
	VP8_BILINEAR        = 2
	VP8_NONE            = 3
)

func decodeVP8Header(r io.Reader) (*VP8Header, error) {
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

	horizontalCode := binary.LittleEndian.Uint16(header[8:10])
	verticalCode := binary.LittleEndian.Uint16(header[10:12])

	width := horizontalCode & 0x3FFF
	xscale := horizontalCode >> 14

	height := verticalCode & 0x3FFF
	yscale := verticalCode >> 14

	h := &VP8Header{
		Version:  uint8(version),
		PartSize: partSize,
		Width:    width,
		XScale:   xscale,
		Height:   height,
		YScale:   yscale,
	}

	return h, nil
}

type Colorspace uint8

const (
	CS_YUV      Colorspace = 0
	CS_RESERVED Colorspace = 1
)

type Clamp uint8

const (
	CLAMP_YES Clamp = 0
	CLAMP_NO  Clamp = 1
)

type SegmentMode uint8

const (
	SEGMODE_DELTA    SegmentMode = 0
	SEGMODE_ABSOLUTE SegmentMode = 1
)

type FilterType uint8

const (
	FILTERTYPE_DISABLED FilterType = 0
	FILTERTYPE_ENABLED  FilterType = 1
)

type VP8FrameHeader struct {
	Colorspace   Colorspace
	ClampingType Clamp

	UseSegment  bool
	UpdateMap   bool
	UpdateData  bool
	SegmentMode SegmentMode
	Quantizer   [4]int8
	LoopFilters [4]int8
	ProbSegment [3]uint8

	FilterType  FilterType
	FilterLevel uint8
	Sharpness   uint8

	FilterAdjustments bool
	FilterUpdateDelta bool
	FilterFrameDelta  [4]int8
	FilterModeDelta   [4]int8

	Partitions uint8

	BaseQ   uint8
	Q_Y1_DC int8
	Q_Y2_DC int8
	Q_Y2_AC int8
	Q_UV_DC int8
	Q_UV_AC int8

	RefreshEntropy bool

	TokenProbUpdate []uint8
	NoSkipCoeff     bool
	ProbSkipFalse   uint8
}

func decodeVP8Frame(r io.Reader, size int) (*VP8FrameHeader, error) {
	frame := make([]byte, size)
	n, err := r.Read(frame)
	if err != nil {
		return nil, NewFormatErrorf("error reading VP8 frame: %v", err)
	}
	if n != size {
		return nil, NewFormatErrorf("invalid VP8 frame: expected %d bytes, got %d", size, n)
	}

	decoder := compression.NewDecoder(frame)
	if decoder == nil {
		return nil, NewFormatError("invalid VP8 frame")
	}

	fc := &VP8FrameHeader{}

	fc.Colorspace = Colorspace(decoder.Read8(1))
	fc.ClampingType = Clamp(decoder.Read8(1))

	fc.UseSegment = decoder.ReadFlag()
	if fc.UseSegment {
		decodeVP8FrameSegment(decoder, fc)
	}

	fc.FilterLevel = decoder.Read8(1)
	fc.FilterLevel = decoder.Read8(6)
	fc.Sharpness = decoder.Read8(3)

	fc.FilterAdjustments = decoder.ReadFlag()
	if fc.FilterAdjustments {
		decodeVP8FilterAdjustments(decoder, fc)
	}

	fc.Partitions = partitionTable(decoder.Read8(2))
	if fc.Partitions == 0 {
		return nil, NewFormatErrorf("invalid partition table, expected 1, 2, 4 or 8 partitions, got %d", fc.Partitions)
	}

	decodeVP8QuantIndices(decoder, fc)

	fc.RefreshEntropy = decoder.ReadFlag()

	tokenProbUpdate(decoder, fc)

	fc.NoSkipCoeff = decoder.ReadFlag()
	if fc.NoSkipCoeff {
		fc.ProbSkipFalse = decoder.Read8(8)
	}

	fmt.Printf("%+v\n", fc)
	fmt.Printf("%+v\n", size)

	return fc, nil
}

// partitions = 2^coef; where coef <= 3
func partitionTable(coef uint8) uint8 {
	switch coef {
	case 0b00:
		return 1
	case 0b01:
		return 2
	case 0b10:
		return 4
	case 0b11:
		return 8
	}

	return 0
}

func decodeVP8FrameSegment(decoder *compression.Decoder, fc *VP8FrameHeader) {
	fc.UpdateMap = decoder.ReadFlag()
	fc.UpdateData = decoder.ReadFlag()

	if fc.UpdateData {
		fc.SegmentMode = SegmentMode(decoder.Read8(1))

		fc.Quantizer = [4]int8{}
		for i := range 4 {
			if decoder.ReadFlag() {
				fc.Quantizer[i] = readSigned(decoder, 7)
			}
		}

		fc.LoopFilters = [4]int8{}
		for i := range 4 {
			if decoder.ReadFlag() {
				fc.LoopFilters[i] = readSigned(decoder, 6)
			}
		}
	}

	if fc.UpdateMap {
		for i := range 3 {
			if decoder.ReadFlag() {
				fc.ProbSegment[i] = decoder.Read8(8)
			}
		}
	}
}

func decodeVP8FilterAdjustments(decoder *compression.Decoder, fc *VP8FrameHeader) {
	fc.FilterUpdateDelta = decoder.ReadFlag()
	if fc.FilterUpdateDelta {
		for i := range 4 {
			if decoder.ReadFlag() {
				fc.FilterFrameDelta[i] = readSigned(decoder, 6)
			}
		}

		for i := range 4 {
			if decoder.ReadFlag() {
				fc.FilterModeDelta[i] = readSigned(decoder, 6)
			}
		}
	}
}

func decodeVP8QuantIndices(decoder *compression.Decoder, fc *VP8FrameHeader) {
	fc.BaseQ = decoder.Read8(7)

	if decoder.ReadFlag() {
		fc.Q_Y1_DC = readSigned(decoder, 4)
	}

	if decoder.ReadFlag() {
		fc.Q_Y2_DC = readSigned(decoder, 4)
	}
	if decoder.ReadFlag() {
		fc.Q_Y2_AC = readSigned(decoder, 4)
	}

	if decoder.ReadFlag() {
		fc.Q_UV_DC = readSigned(decoder, 4)
	}
	if decoder.ReadFlag() {
		fc.Q_UV_AC = readSigned(decoder, 4)
	}
}

func readSigned(decoder *compression.Decoder, l int) int8 {
	v := decoder.Read8(l)

	if !decoder.ReadFlag() {
		return int8(v)
	}

	return -int8(v)
}

func tokenProbUpdate(decoder *compression.Decoder, fc *VP8FrameHeader) {
	fc.TokenProbUpdate = make([]byte, 1056)

	for i := range 1056 {
		if decoder.ReadFlag() {
			fc.TokenProbUpdate[i] = decoder.Read8(8)
		}
	}
}
