package gowebp

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"

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

type decoder struct {
	first      *compression.Decoder
	partitions [8]*compression.Decoder

	r io.Reader

	vp8 *VP8Header
	fc  *VP8FrameHeader
}

func Decode(r io.Reader) (*RIFFHeader, *VP8Header, *VP8FrameHeader, error) {
	decoder := decoder{r: r}

	riffHeader, err := decodeRiffHeader(r)
	if err != nil {
		return nil, nil, nil, err
	}

	vp8Header, err := decoder.decodeVP8Header(r)
	if err != nil {
		return nil, nil, nil, err
	}

	decoder.vp8 = vp8Header

	frame := make([]byte, vp8Header.PartSize)
	n, err := r.Read(frame)
	if err != nil {
		return nil, nil, nil, NewFormatErrorf("error reading VP8 frame: %v", err)
	}
	if n != int(vp8Header.PartSize) {
		return nil, nil, nil, NewFormatErrorf("invalid VP8 frame: expected %d bytes, got %d", int(vp8Header.PartSize), n)
	}
	d := compression.NewDecoder(frame)
	if d == nil {
		return nil, nil, nil, NewFormatError("invalid VP8 frame")
	}
	decoder.first = d

	decoder.vp8.TotalSize = 20 - vp8Header.PartSize

	vp8FrameHeader, err := decoder.decodeVP8Frame()
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
	Version   uint8
	PartSize  uint32
	TotalSize uint32

	Partitions [][]byte

	Width  uint16
	XScale uint16
	Height uint16
	YScale uint16

	MBWidth  uint16
	MBHeight uint16
}

const (
	VP8_BICUBIC         = 0
	VP8_BILINEAR_SIMPLE = 1
	VP8_BILINEAR        = 2
	VP8_NONE            = 3
)

func (d *decoder) decodeVP8Header(r io.Reader) (*VP8Header, error) {
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

	mbw := (width + 0x0f) >> 4
	mbh := (height + 0x0f) >> 4

	h := &VP8Header{
		Version:  uint8(version),
		PartSize: partSize,
		Width:    width,
		XScale:   xscale,
		Height:   height,
		YScale:   yscale,
		MBWidth:  mbw,
		MBHeight: mbh,
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

	NoSkipCoeff   bool
	ProbSkipFalse uint8

	CoeffProbs [4][8][3][11]uint8

	MBTop       []mb
	MBLeft      mb
	Macroblocks []*MacroBlock
}

func (d *decoder) decodeVP8Frame() (*VP8FrameHeader, error) {
	fc := &VP8FrameHeader{}
	d.fc = fc

	fc.Colorspace = Colorspace(d.first.Read8(1))
	fc.ClampingType = Clamp(d.first.Read8(1))

	fc.UseSegment = d.first.ReadFlag()
	if fc.UseSegment {
		d.decodeVP8FrameSegment()
	}

	fc.FilterLevel = d.first.Read8(1)
	fc.FilterLevel = d.first.Read8(6)
	fc.Sharpness = d.first.Read8(3)

	fc.FilterAdjustments = d.first.ReadFlag()
	if fc.FilterAdjustments {
		d.decodeVP8FilterAdjustments()
	}

	fc.Partitions = partitionTable(d.first.Read8(2))
	if fc.Partitions == 0 {
		return nil, NewFormatErrorf("invalid partition table, expected 1, 2, 4 or 8 partitions, got %d", fc.Partitions)
	}
	partLen := [8]int{}
	partadd := 3 * (fc.Partitions - 1) // additional 3 bytes for each part length
	partLen[fc.Partitions-1] = int(d.vp8.TotalSize) - int(partadd)
	if partadd > 0 {
		buf := make([]byte, partadd)
		_, err := d.r.Read(buf)
		if err != nil {
			return nil, err
		}

		for i := range fc.Partitions - 1 {
			pl := int(buf[3*i]) | int(buf[3*i+1])<<8 | int(buf[3*i+2])<<16
			partLen[i] = pl
			partLen[fc.Partitions-1] -= pl
		}
	}

	buf := make([]byte, d.vp8.TotalSize-uint32(partadd))
	_, err := d.r.Read(buf)
	if err != nil {
		return nil, err
	}

	for i, pl := range partLen {
		if i == int(d.fc.Partitions) {
			break
		}

		d.partitions[i] = compression.NewDecoder(buf[:pl])
		buf = buf[pl:]
	}

	d.decodeVP8QuantIndices()

	fc.RefreshEntropy = d.first.ReadFlag()

	d.tokenProbUpdate()

	fc.NoSkipCoeff = d.first.ReadFlag()
	if fc.NoSkipCoeff {
		fc.ProbSkipFalse = d.first.Read8(8)
	}

	fc.MBTop = make([]mb, d.vp8.MBWidth)

	for y := range d.vp8.MBHeight {
		fc.MBLeft = mb{}
		for x := range d.vp8.MBWidth {
			d.recontruct(int(x), int(y))
		}
	}

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

func (d *decoder) decodeVP8FrameSegment() {
	d.fc.UpdateMap = d.first.ReadFlag()
	d.fc.UpdateData = d.first.ReadFlag()

	if d.fc.UpdateData {
		d.fc.SegmentMode = SegmentMode(d.first.Read8(1))

		d.fc.Quantizer = [4]int8{}
		for i := range 4 {
			if d.first.ReadFlag() {
				d.fc.Quantizer[i] = d.readSigned(d.first, 7)
			}
		}

		d.fc.LoopFilters = [4]int8{}
		for i := range 4 {
			if d.first.ReadFlag() {
				d.fc.LoopFilters[i] = d.readSigned(d.first, 6)
			}
		}
	}

	if d.fc.UpdateMap {
		for i := range 3 {
			if d.first.ReadFlag() {
				d.fc.ProbSegment[i] = d.first.Read8(8)
			}
		}
	}
}

func (d *decoder) decodeVP8FilterAdjustments() {
	d.fc.FilterUpdateDelta = d.first.ReadFlag()
	if d.fc.FilterUpdateDelta {
		for i := range 4 {
			if d.first.ReadFlag() {
				d.fc.FilterFrameDelta[i] = d.readSigned(d.first, 6)
			}
		}

		for i := range 4 {
			if d.first.ReadFlag() {
				d.fc.FilterModeDelta[i] = d.readSigned(d.first, 6)
			}
		}
	}
}

func (d *decoder) decodeVP8QuantIndices() {
	d.fc.BaseQ = d.first.Read8(7)

	if d.first.ReadFlag() {
		d.fc.Q_Y1_DC = d.readSigned(d.first, 4)
	}

	if d.first.ReadFlag() {
		d.fc.Q_Y2_DC = d.readSigned(d.first, 4)
	}
	if d.first.ReadFlag() {
		d.fc.Q_Y2_AC = d.readSigned(d.first, 4)
	}

	if d.first.ReadFlag() {
		d.fc.Q_UV_DC = d.readSigned(d.first, 4)
	}
	if d.first.ReadFlag() {
		d.fc.Q_UV_AC = d.readSigned(d.first, 4)
	}
}

func (d *decoder) readSigned(decoder *compression.Decoder, l int) int8 {
	v := decoder.Read8(l)

	if !decoder.ReadFlag() {
		return int8(v)
	}

	return -int8(v)
}

func (d *decoder) tokenProbUpdate() {
	for i := range 4 {
		for j := range 8 {
			for k := range 3 {
				for t := range 11 {
					if d.first.ReadFlagProb(coeffUpdateProbs[i][j][k][t]) {
						d.fc.CoeffProbs[i][j][k][t] = d.first.Read8(8)
					} else { // there is only a single frame so defaults can be set here safely
						d.fc.CoeffProbs[i][j][k][t] = defaultCoeffProbs[i][j][k][t]
					}
				}
			}
		}
	}
}

type MacroBlock struct {
	SegmentId int8
	SkipCoef  bool

	Luma       int8
	LumaBModes [4][4]int8

	Chroma int8

	Coeffs [400]uint16
}

type mb struct {
	pred   [4]int8
	nzMask uint8
	nzY16  uint8
}

func (d *decoder) recontruct(x, y int) *MacroBlock {
	mb := &MacroBlock{}

	if d.fc.UseSegment && d.fc.UpdateMap {
		mb.SegmentId = decodeTree(d.first, mbSegmentTree[:], d.fc.ProbSegment[:])
	}

	if d.fc.NoSkipCoeff {
		mb.SkipCoef = d.first.ReadFlagProb(d.fc.ProbSkipFalse)
	}

	mb.Luma = decodeTree(d.first, kfYmodeTree[:], kfYmodeProb[:])

	mb.LumaBModes = [4][4]int8{}
	switch mb.Luma {
	case B_PRED:
		for j := range 4 {
			a := int8(d.fc.MBLeft.pred[j])
			for i := range 4 {
				if i > 0 {
					a = mb.LumaBModes[j][i-1]
				}
				l := int8(d.fc.MBTop[x].pred[i])
				if j > 0 {
					l = mb.LumaBModes[j-1][i]
				}
				mb.LumaBModes[j][i] = decodeTree(d.first, bmodeTree[:], kfBmodeProb[l][a][:])
			}
		}
	case DC_PRED:
		for j := range 4 {
			for i := range 4 {
				mb.LumaBModes[j][i] = B_DC_PRED
			}
		}
	case V_PRED:
		for j := range 4 {
			for i := range 4 {
				mb.LumaBModes[j][i] = B_VE_PRED
			}
		}
	case H_PRED:
		for j := range 4 {
			for i := range 4 {
				mb.LumaBModes[j][i] = B_HE_PRED
			}
		}
	case TM_PRED:
		for j := range 4 {
			for i := range 4 {
				mb.LumaBModes[j][i] = B_TM_PRED
			}
		}
	}

	d.fc.MBLeft.pred = [4]int8{
		mb.LumaBModes[0][0],
		mb.LumaBModes[0][1],
		mb.LumaBModes[0][2],
		mb.LumaBModes[0][3],
	}
	d.fc.MBTop[x].pred = [4]int8{
		mb.LumaBModes[0][3],
		mb.LumaBModes[1][3],
		mb.LumaBModes[2][3],
		mb.LumaBModes[3][3],
	}

	mb.Chroma = decodeTree(d.first, uvModeTree[:], uvModeProb[:])

	base := 0

	yPlane := PLANE_Y0
	if !mb.SkipCoef {
		coefP := d.partitions[y&(int(d.fc.Partitions)-1)]

		if mb.Luma != B_PRED {
			ctx := d.fc.MBLeft.nzY16 + d.fc.MBTop[x].nzY16
			v := d.decodeToken(coefP, PLANE_Y2, ctx, base, mb)
			base += 16
			if v {
				d.fc.MBLeft.nzY16 = 1
				d.fc.MBTop[x].nzY16 = 1
			} else {
				d.fc.MBLeft.nzY16 = 0
				d.fc.MBTop[x].nzY16 = 0
			}
			yPlane = PLANE_Y1
		}

		lnz := unpack[d.fc.MBLeft.nzMask&0x0f]
		unz := unpack[d.fc.MBTop[x].nzMask&0x0f]
		for j := range 4 {
			nz := lnz[j]
			for i := range 4 {
				v := d.decodeToken(coefP, yPlane, nz+unz[i], base, mb)
				base += 16
				if v {
					unz[i], nz = 1, 1
				} else {
					unz[i], nz = 0, 0
				}
			}
			lnz[j] = nz
		}
		lnzMask := lnz[0] | lnz[1]<<1 | lnz[2]<<2 | lnz[3]<<3
		unzMask := unz[0] | unz[1]<<1 | unz[2]<<2 | unz[3]<<3

		lnz = unpack[d.fc.MBLeft.nzMask>>4]
		unz = unpack[d.fc.MBTop[x].nzMask>>4]
		for c := 0; c < 4; c += 2 {
			for j := range 2 {
				nz := lnz[j+c]
				for i := range 2 {
					v := d.decodeToken(coefP, PLANE_UorV, nz+unz[i+c], base, mb)
					base += 16
					if v {
						unz[i], nz = 1, 1
					} else {
						unz[i], nz = 0, 0
					}
				}
				lnz[j+c] = nz
			}
		}
		lnzMask |= (lnz[0] | lnz[1]<<1 | lnz[2]<<2 | lnz[3]<<3) << 4
		unzMask |= (unz[0] | unz[1]<<1 | unz[2]<<2 | unz[3]<<3) << 4

		d.fc.MBLeft.nzMask = lnzMask
		d.fc.MBTop[x].nzMask = unzMask
	} else {
		if mb.Luma != B_PRED {
			d.fc.MBLeft.nzY16 = 0
			d.fc.MBTop[x].nzY16 = 0
		}

		d.fc.MBLeft.nzMask = 0
		d.fc.MBTop[x].nzMask = 0
	}

	d.fc.Macroblocks = append(d.fc.Macroblocks, mb)
	return mb
}

func (d *decoder) dctExtra(decoder *compression.Decoder, prob []uint8) int8 {
	pos := 0
	v := decoder.Read8Prob(1, prob[pos])
	pos++

	for ; prob[pos] > 0; pos++ {
		v += decoder.Read8Prob(1, prob[pos])
	}

	iv := int8(v)
	if iv != 0 && decoder.ReadFlag() {
		iv = -iv
	}

	return iv
}

func (d *decoder) decodeToken(decoder *compression.Decoder, plane, context uint8, base int, mb *MacroBlock) bool {
	block := [16]int{}
	var firstCoeff, ctx2 int
	probTable := [11]uint8{}
	var token, extraBits int8
	var absValue int

	prevCoeffZero := false
	currentBlockCoeffs := false

	categoryBase := [6]uint8{5, 7, 11, 19, 35, 67}

	if plane == 1 {
		firstCoeff++
	}

	for i := firstCoeff; i < 16; i++ {
		ctx2 = coeffBands[i]
		probTable = d.fc.CoeffProbs[plane][ctx2][context]

		if prevCoeffZero {
			token = decodeTree(decoder, coeffTreeNoEOB[:], probTable[:])
		} else {
			token = decodeTree(decoder, coeffTree[:], probTable[:])
		}

		if token == dct_eob {
			break
		}

		if token != DCT_0 {
			currentBlockCoeffs = true

			if token >= dct_cat1 && token <= dct_cat6 {
				catbase := int8(math.Abs(float64(token))) - dct_cat1
				extraBits = d.dctExtra(decoder, pCatBases[catbase])
				absValue = int(categoryBase[catbase]) + int(extraBits)
			} else {
				absValue = int(math.Abs(float64(token)))
			}

			if decoder.ReadFlag() {
				block[i] = -absValue
			} else {
				block[i] = absValue
			}

			mb.Coeffs[base+int(zigzag[i])] = uint16(block[i])
		} else {
			absValue = int(math.Abs(float64(token)))
		}

		if absValue == 0 {
			context = 0
		} else if absValue == 1 {
			context = 1
		} else {
			context = 2
		}
		prevCoeffZero = true
	}

	return currentBlockCoeffs
}

func decodeTree(decoder *compression.Decoder, tree []int8, p []uint8) int8 {
	i := tree[int8(decoder.Read8Prob(1, p[0]))]

	for i > 0 {
		i = tree[i+int8(decoder.Read8Prob(1, p[i>>1]))]
	}

	return -i
}

var mbSegmentTree = [6]int8{
	2, 4,
	-0, -1,
	-2, -3,
}

const (
	DC_PRED int8 = iota
	V_PRED
	H_PRED
	TM_PRED

	B_PRED
)

var kfYmodeTree = [8]int8{
	-B_PRED, 2,
	4, 6,
	-DC_PRED, -V_PRED,
	-H_PRED, -TM_PRED,
}

var kfYmodeProb = [4]uint8{145, 156, 163, 128}

const (
	B_DC_PRED int8 = iota
	B_TM_PRED

	B_VE_PRED
	B_HE_PRED

	B_LD_PRED
	B_RD_PRED

	B_VR_PRED
	B_VL_PRED
	B_HD_PRED
	B_HU_PRED
)

var bmodeTree = [18]int8{
	-B_DC_PRED, 2,
	-B_TM_PRED, 4,
	-B_VE_PRED, 6,
	8, 12,
	-B_HE_PRED, 10,
	-B_RD_PRED, -B_VR_PRED,
	-B_LD_PRED, 14,
	-B_VL_PRED, 16,
	-B_HD_PRED, -B_HU_PRED,
}

var kfBmodeProb = [10][10][9]uint8{
	{
		{231, 120, 48, 89, 115, 113, 120, 152, 112},
		{152, 179, 64, 126, 170, 118, 46, 70, 95},
		{175, 69, 143, 80, 85, 82, 72, 155, 103},
		{56, 58, 10, 171, 218, 189, 17, 13, 152},
		{144, 71, 10, 38, 171, 213, 144, 34, 26},
		{114, 26, 17, 163, 44, 195, 21, 10, 173},
		{121, 24, 80, 195, 26, 62, 44, 64, 85},
		{170, 46, 55, 19, 136, 160, 33, 206, 71},
		{63, 20, 8, 114, 114, 208, 12, 9, 226},
		{81, 40, 11, 96, 182, 84, 29, 16, 36},
	},
	{
		{134, 183, 89, 137, 98, 101, 106, 165, 148},
		{72, 187, 100, 130, 157, 111, 32, 75, 80},
		{66, 102, 167, 99, 74, 62, 40, 234, 128},
		{41, 53, 9, 178, 241, 141, 26, 8, 107},
		{104, 79, 12, 27, 217, 255, 87, 17, 7},
		{74, 43, 26, 146, 73, 166, 49, 23, 157},
		{65, 38, 105, 160, 51, 52, 31, 115, 128},
		{87, 68, 71, 44, 114, 51, 15, 186, 23},
		{47, 41, 14, 110, 182, 183, 21, 17, 194},
		{66, 45, 25, 102, 197, 189, 23, 18, 22},
	},
	{
		{88, 88, 147, 150, 42, 46, 45, 196, 205},
		{43, 97, 183, 117, 85, 38, 35, 179, 61},
		{39, 53, 200, 87, 26, 21, 43, 232, 171},
		{56, 34, 51, 104, 114, 102, 29, 93, 77},
		{107, 54, 32, 26, 51, 1, 81, 43, 31},
		{39, 28, 85, 171, 58, 165, 90, 98, 64},
		{34, 22, 116, 206, 23, 34, 43, 166, 73},
		{68, 25, 106, 22, 64, 171, 36, 225, 114},
		{34, 19, 21, 102, 132, 188, 16, 76, 124},
		{62, 18, 78, 95, 85, 57, 50, 48, 51},
	},
	{
		{193, 101, 35, 159, 215, 111, 89, 46, 111},
		{60, 148, 31, 172, 219, 228, 21, 18, 111},
		{112, 113, 77, 85, 179, 255, 38, 120, 114},
		{40, 42, 1, 196, 245, 209, 10, 25, 109},
		{100, 80, 8, 43, 154, 1, 51, 26, 71},
		{88, 43, 29, 140, 166, 213, 37, 43, 154},
		{61, 63, 30, 155, 67, 45, 68, 1, 209},
		{142, 78, 78, 16, 255, 128, 34, 197, 171},
		{41, 40, 5, 102, 211, 183, 4, 1, 221},
		{51, 50, 17, 168, 209, 192, 23, 25, 82},
	},
	{
		{125, 98, 42, 88, 104, 85, 117, 175, 82},
		{95, 84, 53, 89, 128, 100, 113, 101, 45},
		{75, 79, 123, 47, 51, 128, 81, 171, 1},
		{57, 17, 5, 71, 102, 57, 53, 41, 49},
		{115, 21, 2, 10, 102, 255, 166, 23, 6},
		{38, 33, 13, 121, 57, 73, 26, 1, 85},
		{41, 10, 67, 138, 77, 110, 90, 47, 114},
		{101, 29, 16, 10, 85, 128, 101, 196, 26},
		{57, 18, 10, 102, 102, 213, 34, 20, 43},
		{117, 20, 15, 36, 163, 128, 68, 1, 26},
	},
	{
		{138, 31, 36, 171, 27, 166, 38, 44, 229},
		{67, 87, 58, 169, 82, 115, 26, 59, 179},
		{63, 59, 90, 180, 59, 166, 93, 73, 154},
		{40, 40, 21, 116, 143, 209, 34, 39, 175},
		{57, 46, 22, 24, 128, 1, 54, 17, 37},
		{47, 15, 16, 183, 34, 223, 49, 45, 183},
		{46, 17, 33, 183, 6, 98, 15, 32, 183},
		{65, 32, 73, 115, 28, 128, 23, 128, 205},
		{40, 3, 9, 115, 51, 192, 18, 6, 223},
		{87, 37, 9, 115, 59, 77, 64, 21, 47},
	},
	{
		{104, 55, 44, 218, 9, 54, 53, 130, 226},
		{64, 90, 70, 205, 40, 41, 23, 26, 57},
		{54, 57, 112, 184, 5, 41, 38, 166, 213},
		{30, 34, 26, 133, 152, 116, 10, 32, 134},
		{75, 32, 12, 51, 192, 255, 160, 43, 51},
		{39, 19, 53, 221, 26, 114, 32, 73, 255},
		{31, 9, 65, 234, 2, 15, 1, 118, 73},
		{88, 31, 35, 67, 102, 85, 55, 186, 85},
		{56, 21, 23, 111, 59, 205, 45, 37, 192},
		{55, 38, 70, 124, 73, 102, 1, 34, 98},
	},
	{
		{102, 61, 71, 37, 34, 53, 31, 243, 192},
		{69, 60, 71, 38, 73, 119, 28, 222, 37},
		{68, 45, 128, 34, 1, 47, 11, 245, 171},
		{62, 17, 19, 70, 146, 85, 55, 62, 70},
		{75, 15, 9, 9, 64, 255, 184, 119, 16},
		{37, 43, 37, 154, 100, 163, 85, 160, 1},
		{63, 9, 92, 136, 28, 64, 32, 201, 85},
		{86, 6, 28, 5, 64, 255, 25, 248, 1},
		{56, 8, 17, 132, 137, 255, 55, 116, 128},
		{58, 15, 20, 82, 135, 57, 26, 121, 40},
	},
	{
		{164, 50, 31, 137, 154, 133, 25, 35, 218},
		{51, 103, 44, 131, 131, 123, 31, 6, 158},
		{86, 40, 64, 135, 148, 224, 45, 183, 128},
		{22, 26, 17, 131, 240, 154, 14, 1, 209},
		{83, 12, 13, 54, 192, 255, 68, 47, 28},
		{45, 16, 21, 91, 64, 222, 7, 1, 197},
		{56, 21, 39, 155, 60, 138, 23, 102, 213},
		{85, 26, 85, 85, 128, 128, 32, 146, 171},
		{18, 11, 7, 63, 144, 171, 4, 4, 246},
		{35, 27, 10, 146, 174, 171, 12, 26, 128},
	},
	{
		{190, 80, 35, 99, 180, 80, 126, 54, 45},
		{85, 126, 47, 87, 176, 51, 41, 20, 32},
		{101, 75, 128, 139, 118, 146, 116, 128, 85},
		{56, 41, 15, 176, 236, 85, 37, 9, 62},
		{146, 36, 19, 30, 171, 255, 97, 27, 20},
		{71, 30, 17, 119, 118, 255, 17, 18, 138},
		{101, 38, 60, 138, 55, 70, 43, 26, 142},
		{138, 45, 61, 62, 219, 1, 81, 188, 64},
		{32, 41, 20, 117, 151, 142, 20, 21, 163},
		{112, 19, 12, 61, 195, 128, 48, 4, 24},
	},
}

var uvModeTree = [6]int8{
	-DC_PRED, 2,
	-V_PRED, 4,
	-H_PRED, -TM_PRED,
}
var uvModeProb = [3]uint8{142, 114, 183}

const (
	DCT_0 int8 = iota
	DCT_1
	DCT_2
	DCT_3
	DCT_4
	dct_cat1
	dct_cat2
	dct_cat3
	dct_cat4
	dct_cat5
	dct_cat6
	dct_eob
)

var coeffTree = [22]int8{
	-dct_eob, 2,
	-DCT_0, 4,
	-DCT_1, 6,
	8, 12,
	-DCT_2, 10,
	-DCT_3, -DCT_4,
	14, 16,
	-dct_cat1, -dct_cat2,
	18, 20,
	-dct_cat3, -dct_cat4,
	-dct_cat5, -dct_cat6,
}

var coeffTreeNoEOB = [20]int8{
	-DCT_0, 4,
	-DCT_1, 6,
	8, 12,
	-DCT_2, 10,
	-DCT_3, -DCT_4,
	14, 16,
	-dct_cat1, -dct_cat2,
	18, 20,
	-dct_cat3, -dct_cat4,
	-dct_cat5, -dct_cat6,
}

var Pcat1 = [2]uint8{159, 0}
var Pcat2 = [3]uint8{165, 145, 0}
var Pcat3 = [4]uint8{173, 148, 140, 0}
var Pcat4 = [5]uint8{176, 155, 140, 135, 0}
var Pcat5 = [6]uint8{180, 157, 141, 134, 130, 0}
var Pcat6 = [12]uint8{254, 254, 243, 230, 196, 177, 153, 140, 133, 130, 129, 0}

var pCatBases = [6][]uint8{
	Pcat1[:],
	Pcat2[:],
	Pcat3[:],
	Pcat4[:],
	Pcat5[:],
	Pcat6[:],
}

var coeffBands = [16]int{0, 1, 2, 3, 6, 4, 5, 6, 6, 6, 6, 6, 6, 6, 6, 7}

const (
	PLANE_Y1 uint8 = iota
	PLANE_Y2
	PLANE_UorV
	PLANE_Y0
)

var coeffUpdateProbs = [4][8][3][11]uint8{
	{
		{
			{255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
			{255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
			{255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
		},
		{
			{176, 246, 255, 255, 255, 255, 255, 255, 255, 255, 255},
			{223, 241, 252, 255, 255, 255, 255, 255, 255, 255, 255},
			{249, 253, 253, 255, 255, 255, 255, 255, 255, 255, 255},
		},
		{
			{255, 244, 252, 255, 255, 255, 255, 255, 255, 255, 255},
			{234, 254, 254, 255, 255, 255, 255, 255, 255, 255, 255},
			{253, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
		},
		{
			{255, 246, 254, 255, 255, 255, 255, 255, 255, 255, 255},
			{239, 253, 254, 255, 255, 255, 255, 255, 255, 255, 255},
			{254, 255, 254, 255, 255, 255, 255, 255, 255, 255, 255},
		},
		{
			{255, 248, 254, 255, 255, 255, 255, 255, 255, 255, 255},
			{251, 255, 254, 255, 255, 255, 255, 255, 255, 255, 255},
			{255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
		},
		{
			{255, 253, 254, 255, 255, 255, 255, 255, 255, 255, 255},
			{251, 254, 254, 255, 255, 255, 255, 255, 255, 255, 255},
			{254, 255, 254, 255, 255, 255, 255, 255, 255, 255, 255},
		},
		{
			{255, 254, 253, 255, 254, 255, 255, 255, 255, 255, 255},
			{250, 255, 254, 255, 254, 255, 255, 255, 255, 255, 255},
			{254, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
		},
		{
			{255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
			{255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
			{255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
		},
	},
	{
		{
			{217, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
			{225, 252, 241, 253, 255, 255, 254, 255, 255, 255, 255},
			{234, 250, 241, 250, 253, 255, 253, 254, 255, 255, 255},
		},
		{
			{255, 254, 255, 255, 255, 255, 255, 255, 255, 255, 255},
			{223, 254, 254, 255, 255, 255, 255, 255, 255, 255, 255},
			{238, 253, 254, 254, 255, 255, 255, 255, 255, 255, 255},
		},
		{
			{255, 248, 254, 255, 255, 255, 255, 255, 255, 255, 255},
			{249, 254, 255, 255, 255, 255, 255, 255, 255, 255, 255},
			{255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
		},
		{
			{255, 253, 255, 255, 255, 255, 255, 255, 255, 255, 255},
			{247, 254, 255, 255, 255, 255, 255, 255, 255, 255, 255},
			{255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
		},
		{
			{255, 253, 254, 255, 255, 255, 255, 255, 255, 255, 255},
			{252, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
			{255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
		},
		{
			{255, 254, 254, 255, 255, 255, 255, 255, 255, 255, 255},
			{253, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
			{255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
		},
		{
			{255, 254, 253, 255, 255, 255, 255, 255, 255, 255, 255},
			{250, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
			{254, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
		},
		{
			{255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
			{255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
			{255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
		},
	},
	{
		{
			{186, 251, 250, 255, 255, 255, 255, 255, 255, 255, 255},
			{234, 251, 244, 254, 255, 255, 255, 255, 255, 255, 255},
			{251, 251, 243, 253, 254, 255, 254, 255, 255, 255, 255},
		},
		{
			{255, 253, 254, 255, 255, 255, 255, 255, 255, 255, 255},
			{236, 253, 254, 255, 255, 255, 255, 255, 255, 255, 255},
			{251, 253, 253, 254, 254, 255, 255, 255, 255, 255, 255},
		},
		{
			{255, 254, 254, 255, 255, 255, 255, 255, 255, 255, 255},
			{254, 254, 254, 255, 255, 255, 255, 255, 255, 255, 255},
			{255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
		},
		{
			{255, 254, 255, 255, 255, 255, 255, 255, 255, 255, 255},
			{254, 254, 255, 255, 255, 255, 255, 255, 255, 255, 255},
			{254, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
		},
		{
			{255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
			{254, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
			{255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
		},
		{
			{255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
			{255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
			{255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
		},
		{
			{255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
			{255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
			{255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
		},
		{
			{255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
			{255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
			{255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
		},
	},
	{
		{
			{248, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
			{250, 254, 252, 254, 255, 255, 255, 255, 255, 255, 255},
			{248, 254, 249, 253, 255, 255, 255, 255, 255, 255, 255},
		},
		{
			{255, 253, 253, 255, 255, 255, 255, 255, 255, 255, 255},
			{246, 253, 253, 255, 255, 255, 255, 255, 255, 255, 255},
			{252, 254, 251, 254, 254, 255, 255, 255, 255, 255, 255},
		},
		{
			{255, 254, 252, 255, 255, 255, 255, 255, 255, 255, 255},
			{248, 254, 253, 255, 255, 255, 255, 255, 255, 255, 255},
			{253, 255, 254, 254, 255, 255, 255, 255, 255, 255, 255},
		},
		{
			{255, 251, 254, 255, 255, 255, 255, 255, 255, 255, 255},
			{245, 251, 254, 255, 255, 255, 255, 255, 255, 255, 255},
			{253, 253, 254, 255, 255, 255, 255, 255, 255, 255, 255},
		},
		{
			{255, 251, 253, 255, 255, 255, 255, 255, 255, 255, 255},
			{252, 253, 254, 255, 255, 255, 255, 255, 255, 255, 255},
			{255, 254, 255, 255, 255, 255, 255, 255, 255, 255, 255},
		},
		{
			{255, 252, 255, 255, 255, 255, 255, 255, 255, 255, 255},
			{249, 255, 254, 255, 255, 255, 255, 255, 255, 255, 255},
			{255, 255, 254, 255, 255, 255, 255, 255, 255, 255, 255},
		},
		{
			{255, 255, 253, 255, 255, 255, 255, 255, 255, 255, 255},
			{250, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
			{255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
		},
		{
			{255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
			{254, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
			{255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
		},
	},
}

var defaultCoeffProbs = [4][8][3][11]uint8{
	{
		{
			{128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
			{128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
			{128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
		},
		{
			{253, 136, 254, 255, 228, 219, 128, 128, 128, 128, 128},
			{189, 129, 242, 255, 227, 213, 255, 219, 128, 128, 128},
			{106, 126, 227, 252, 214, 209, 255, 255, 128, 128, 128},
		},
		{
			{1, 98, 248, 255, 236, 226, 255, 255, 128, 128, 128},
			{181, 133, 238, 254, 221, 234, 255, 154, 128, 128, 128},
			{78, 134, 202, 247, 198, 180, 255, 219, 128, 128, 128},
		},
		{
			{1, 185, 249, 255, 243, 255, 128, 128, 128, 128, 128},
			{184, 150, 247, 255, 236, 224, 128, 128, 128, 128, 128},
			{77, 110, 216, 255, 236, 230, 128, 128, 128, 128, 128},
		},
		{
			{1, 101, 251, 255, 241, 255, 128, 128, 128, 128, 128},
			{170, 139, 241, 252, 236, 209, 255, 255, 128, 128, 128},
			{37, 116, 196, 243, 228, 255, 255, 255, 128, 128, 128},
		},
		{
			{1, 204, 254, 255, 245, 255, 128, 128, 128, 128, 128},
			{207, 160, 250, 255, 238, 128, 128, 128, 128, 128, 128},
			{102, 103, 231, 255, 211, 171, 128, 128, 128, 128, 128},
		},
		{
			{1, 152, 252, 255, 240, 255, 128, 128, 128, 128, 128},
			{177, 135, 243, 255, 234, 225, 128, 128, 128, 128, 128},
			{80, 129, 211, 255, 194, 224, 128, 128, 128, 128, 128},
		},
		{
			{1, 1, 255, 128, 128, 128, 128, 128, 128, 128, 128},
			{246, 1, 255, 128, 128, 128, 128, 128, 128, 128, 128},
			{255, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
		},
	},
	{
		{
			{198, 35, 237, 223, 193, 187, 162, 160, 145, 155, 62},
			{131, 45, 198, 221, 172, 176, 220, 157, 252, 221, 1},
			{68, 47, 146, 208, 149, 167, 221, 162, 255, 223, 128},
		},
		{
			{1, 149, 241, 255, 221, 224, 255, 255, 128, 128, 128},
			{184, 141, 234, 253, 222, 220, 255, 199, 128, 128, 128},
			{81, 99, 181, 242, 176, 190, 249, 202, 255, 255, 128},
		},
		{
			{1, 129, 232, 253, 214, 197, 242, 196, 255, 255, 128},
			{99, 121, 210, 250, 201, 198, 255, 202, 128, 128, 128},
			{23, 91, 163, 242, 170, 187, 247, 210, 255, 255, 128},
		},
		{
			{1, 200, 246, 255, 234, 255, 128, 128, 128, 128, 128},
			{109, 178, 241, 255, 231, 245, 255, 255, 128, 128, 128},
			{44, 130, 201, 253, 205, 192, 255, 255, 128, 128, 128},
		},
		{
			{1, 132, 239, 251, 219, 209, 255, 165, 128, 128, 128},
			{94, 136, 225, 251, 218, 190, 255, 255, 128, 128, 128},
			{22, 100, 174, 245, 186, 161, 255, 199, 128, 128, 128},
		},
		{
			{1, 182, 249, 255, 232, 235, 128, 128, 128, 128, 128},
			{124, 143, 241, 255, 227, 234, 128, 128, 128, 128, 128},
			{35, 77, 181, 251, 193, 211, 255, 205, 128, 128, 128},
		},
		{
			{1, 157, 247, 255, 236, 231, 255, 255, 128, 128, 128},
			{121, 141, 235, 255, 225, 227, 255, 255, 128, 128, 128},
			{45, 99, 188, 251, 195, 217, 255, 224, 128, 128, 128},
		},
		{
			{1, 1, 251, 255, 213, 255, 128, 128, 128, 128, 128},
			{203, 1, 248, 255, 255, 128, 128, 128, 128, 128, 128},
			{137, 1, 177, 255, 224, 255, 128, 128, 128, 128, 128},
		},
	},
	{
		{
			{253, 9, 248, 251, 207, 208, 255, 192, 128, 128, 128},
			{175, 13, 224, 243, 193, 185, 249, 198, 255, 255, 128},
			{73, 17, 171, 221, 161, 179, 236, 167, 255, 234, 128},
		},
		{
			{1, 95, 247, 253, 212, 183, 255, 255, 128, 128, 128},
			{239, 90, 244, 250, 211, 209, 255, 255, 128, 128, 128},
			{155, 77, 195, 248, 188, 195, 255, 255, 128, 128, 128},
		},
		{
			{1, 24, 239, 251, 218, 219, 255, 205, 128, 128, 128},
			{201, 51, 219, 255, 196, 186, 128, 128, 128, 128, 128},
			{69, 46, 190, 239, 201, 218, 255, 228, 128, 128, 128},
		},
		{
			{1, 191, 251, 255, 255, 128, 128, 128, 128, 128, 128},
			{223, 165, 249, 255, 213, 255, 128, 128, 128, 128, 128},
			{141, 124, 248, 255, 255, 128, 128, 128, 128, 128, 128},
		},
		{
			{1, 16, 248, 255, 255, 128, 128, 128, 128, 128, 128},
			{190, 36, 230, 255, 236, 255, 128, 128, 128, 128, 128},
			{149, 1, 255, 128, 128, 128, 128, 128, 128, 128, 128},
		},
		{
			{1, 226, 255, 128, 128, 128, 128, 128, 128, 128, 128},
			{247, 192, 255, 128, 128, 128, 128, 128, 128, 128, 128},
			{240, 128, 255, 128, 128, 128, 128, 128, 128, 128, 128},
		},
		{
			{1, 134, 252, 255, 255, 128, 128, 128, 128, 128, 128},
			{213, 62, 250, 255, 255, 128, 128, 128, 128, 128, 128},
			{55, 93, 255, 128, 128, 128, 128, 128, 128, 128, 128},
		},
		{
			{128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
			{128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
			{128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
		},
	},
	{
		{
			{202, 24, 213, 235, 186, 191, 220, 160, 240, 175, 255},
			{126, 38, 182, 232, 169, 184, 228, 174, 255, 187, 128},
			{61, 46, 138, 219, 151, 178, 240, 170, 255, 216, 128},
		},
		{
			{1, 112, 230, 250, 199, 191, 247, 159, 255, 255, 128},
			{166, 109, 228, 252, 211, 215, 255, 174, 128, 128, 128},
			{39, 77, 162, 232, 172, 180, 245, 178, 255, 255, 128},
		},
		{
			{1, 52, 220, 246, 198, 199, 249, 220, 255, 255, 128},
			{124, 74, 191, 243, 183, 193, 250, 221, 255, 255, 128},
			{24, 71, 130, 219, 154, 170, 243, 182, 255, 255, 128},
		},
		{
			{1, 182, 225, 249, 219, 240, 255, 224, 128, 128, 128},
			{149, 150, 226, 252, 216, 205, 255, 171, 128, 128, 128},
			{28, 108, 170, 242, 183, 194, 254, 223, 255, 255, 128},
		},
		{
			{1, 81, 230, 252, 204, 203, 255, 192, 128, 128, 128},
			{123, 102, 209, 247, 188, 196, 255, 233, 128, 128, 128},
			{20, 95, 153, 243, 164, 173, 255, 203, 128, 128, 128},
		},
		{
			{1, 222, 248, 255, 216, 213, 128, 128, 128, 128, 128},
			{168, 175, 246, 252, 235, 205, 255, 255, 128, 128, 128},
			{47, 116, 215, 255, 211, 212, 255, 255, 128, 128, 128},
		},
		{
			{1, 121, 236, 253, 212, 214, 255, 255, 128, 128, 128},
			{141, 84, 213, 252, 201, 202, 255, 219, 128, 128, 128},
			{42, 80, 160, 240, 162, 185, 255, 205, 128, 128, 128},
		},
		{
			{1, 1, 255, 128, 128, 128, 128, 128, 128, 128, 128},
			{244, 1, 255, 128, 128, 128, 128, 128, 128, 128, 128},
			{238, 1, 255, 128, 128, 128, 128, 128, 128, 128, 128},
		},
	},
}

var unpack = [16][4]uint8{
	{0, 0, 0, 0},
	{1, 0, 0, 0},
	{0, 1, 0, 0},
	{1, 1, 0, 0},
	{0, 0, 1, 0},
	{1, 0, 1, 0},
	{0, 1, 1, 0},
	{1, 1, 1, 0},
	{0, 0, 0, 1},
	{1, 0, 0, 1},
	{0, 1, 0, 1},
	{1, 1, 0, 1},
	{0, 0, 1, 1},
	{1, 0, 1, 1},
	{0, 1, 1, 1},
	{1, 1, 1, 1},
}

var zigzag = [16]uint8{0, 1, 4, 8, 5, 2, 3, 6, 9, 12, 13, 10, 7, 11, 14, 15}
