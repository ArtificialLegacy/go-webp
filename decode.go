package gowebp

import (
	"encoding/binary"
	"fmt"
	"image"
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

type decoder struct {
	first      *compression.Decoder
	partitions [8]*compression.Decoder

	r io.Reader

	vp8      *VP8Header
	fc       *VP8FrameHeader
	nzDCMask uint32
	nzACMask uint32

	img *image.YCbCr
}

func Decode(r io.Reader) (*RIFFHeader, *VP8Header, *VP8FrameHeader, image.Image, error) {
	decoder := decoder{r: r}

	riffHeader, err := decodeRiffHeader(r)
	if err != nil {
		return nil, nil, nil, nil, err
	}

	vp8Header, err := decoder.decodeVP8Header(r)
	if err != nil {
		return nil, nil, nil, nil, err
	}

	decoder.vp8 = vp8Header

	frame := make([]byte, vp8Header.PartSize)
	n, err := r.Read(frame)
	if err != nil {
		return nil, nil, nil, nil, NewFormatErrorf("error reading VP8 frame: %v", err)
	}
	if n != int(vp8Header.PartSize) {
		return nil, nil, nil, nil, NewFormatErrorf("invalid VP8 frame: expected %d bytes, got %d", int(vp8Header.PartSize), n)
	}
	d := compression.NewDecoder(frame)
	if d == nil {
		return nil, nil, nil, nil, NewFormatError("invalid VP8 frame")
	}
	decoder.first = d

	decoder.vp8.TotalSize = 20 - vp8Header.PartSize

	vp8FrameHeader, err := decoder.decodeVP8Frame()
	if err != nil {
		return nil, nil, nil, nil, err
	}

	return riffHeader, vp8Header, vp8FrameHeader, decoder.img, nil
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

	Quant [4]quant

	RefreshEntropy bool

	NoSkipCoeff   bool
	ProbSkipFalse uint8

	CoeffProbs [4][8][3][11]uint8

	MBTop       []mb
	MBLeft      mb
	Macroblocks []*MacroBlock

	YBR [26][32]uint8
}

type quant struct {
	y1 [2]uint16
	y2 [2]uint16
	uv [2]uint16
}

func (d *decoder) decodeVP8Frame() (*VP8FrameHeader, error) {
	m := image.NewYCbCr(image.Rect(0, 0, 16*int(d.vp8.MBWidth), 16*int(d.vp8.MBHeight)), image.YCbCrSubsampleRatio420)
	d.img = m.SubImage(image.Rect(0, 0, int(d.vp8.Width), int(d.vp8.Height))).(*image.YCbCr)

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

	for i := range 4 {
		q := int32(d.fc.BaseQ)
		if d.fc.UseSegment {
			if d.fc.SegmentMode == SEGMODE_DELTA {
				q += int32(d.fc.Quantizer[i])
			} else {
				q = int32(d.fc.Quantizer[i])
			}
		}

		d.fc.Quant[i].y1[0] = dequantTableDC[clip(q+int32(d.fc.Q_Y1_DC), 0, 127)]
		d.fc.Quant[i].y1[1] = dequantTableAC[clip(q, 0, 127)]
		d.fc.Quant[i].y2[0] = dequantTableDC[clip(q+int32(d.fc.Q_Y2_DC), 0, 127)] * 2
		d.fc.Quant[i].y2[1] = dequantTableAC[clip(q+int32(d.fc.Q_Y2_AC), 0, 127)] * 155 / 100
		if d.fc.Quant[i].y2[1] < 8 {
			d.fc.Quant[i].y2[1] = 8
		}

		d.fc.Quant[i].uv[0] = dequantTableDC[clip(q+int32(d.fc.Q_UV_DC), 0, 117)]
		d.fc.Quant[i].uv[1] = dequantTableAC[clip(q+int32(d.fc.Q_UV_AC), 0, 127)]
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
	SegmentId int
	SkipCoef  bool

	UseLuma16  bool
	Luma16     uint8
	LumaBModes [4][4]uint8

	Chroma int8

	Coeffs [400]int16
}

type mb struct {
	pred   [4]uint8
	nzMask uint8
	nzY16  uint8
}

func (d *decoder) lumaMode(x int, mb *MacroBlock) {
	mb.UseLuma16 = d.first.ReadFlagProb(145)
	if mb.UseLuma16 {
		var p uint8
		if !d.first.ReadFlagProb(156) {
			if !d.first.ReadFlagProb(163) {
				p = predDC
			} else {
				p = predVE
			}
		} else if !d.first.ReadFlagProb(128) {
			p = predHE
		} else {
			p = predTM
		}
		for i := 0; i < 4; i++ {
			d.fc.MBTop[x].pred[i] = p
			d.fc.MBLeft.pred[i] = p
		}
		mb.Luma16 = p
	} else {
		for j := range 4 {
			p := d.fc.MBLeft.pred[j]
			for i := range 4 {
				prob := &kfBmodeProb[d.fc.MBTop[x].pred[i]][p]

				if !d.first.ReadFlagProb(prob[0]) {
					p = predDC
				} else if !d.first.ReadFlagProb(prob[1]) {
					p = predTM
				} else if !d.first.ReadFlagProb(prob[2]) {
					p = predVE
				} else if !d.first.ReadFlagProb(prob[3]) {
					if !d.first.ReadFlagProb(prob[4]) {
						p = predHE
					} else if !d.first.ReadFlagProb(prob[5]) {
						p = predRD
					} else {
						p = predVR
					}
				} else if !d.first.ReadFlagProb(prob[6]) {
					p = predLD
				} else if !d.first.ReadFlagProb(prob[7]) {
					p = predVL
				} else if !d.first.ReadFlagProb(prob[8]) {
					p = predHD
				} else {
					p = predHU
				}
				mb.LumaBModes[j][i] = p
				d.fc.MBTop[x].pred[i] = p
			}
			d.fc.MBLeft.pred[j] = p
		}
	}
}

func (d *decoder) recontruct(x, y int) *MacroBlock {
	if y == 0 {
		fmt.Printf("%+v\n", d.fc.MBLeft)
	}

	mb := &MacroBlock{}

	if d.fc.UpdateMap {
		if !d.first.ReadFlagProb(d.fc.ProbSegment[0]) {
			mb.SegmentId = int(d.first.ReadProb(1, d.fc.ProbSegment[1]))
		} else {
			mb.SegmentId = int(d.first.ReadProb(1, d.fc.ProbSegment[2])) + 2
		}
	}

	if d.fc.NoSkipCoeff {
		mb.SkipCoef = d.first.ReadFlagProb(d.fc.ProbSkipFalse)
	}

	d.prepareYBR(x, y)

	d.lumaMode(x, mb)

	if !d.first.ReadFlagProb(142) {
		mb.Chroma = predDC
	} else if !d.first.ReadFlagProb(114) {
		mb.Chroma = predVE
	} else if !d.first.ReadFlagProb(183) {
		mb.Chroma = predHE
	} else {
		mb.Chroma = predTM
	}

	if !mb.SkipCoef {
		coefP := d.partitions[y&(int(d.fc.Partitions)-1)]
		quant := d.fc.Quant[mb.SegmentId]
		yPlane := PLANE_Y0

		if mb.UseLuma16 {
			ctx := d.fc.MBLeft.nzY16 + d.fc.MBTop[x].nzY16
			nz := d.decodeResid(coefP, PLANE_Y2, ctx, 384, quant.y2, mb)
			d.fc.MBLeft.nzY16 = nz
			d.fc.MBTop[x].nzY16 = nz
			d.inverseWHT16(mb)
			yPlane = PLANE_Y1
		}

		base := 0
		var nzDC, nzAC [4]uint8
		var nzDCMask, nzACMask uint32

		lnz := unpack[d.fc.MBLeft.nzMask&0x0f]
		unz := unpack[d.fc.MBTop[x].nzMask&0x0f]
		for j := range 4 {
			nz := lnz[j]
			for i := range 4 {
				nz = d.decodeResid(coefP, yPlane, nz+unz[i], base, quant.y1, mb)
				unz[i] = nz
				nzAC[i] = nz
				nzDC[i] = btou(mb.Coeffs[base] != 0)
				base += 16
			}
			lnz[j] = nz
			nzDCMask |= pack(nzDC, j*4)
			nzACMask |= pack(nzAC, j*4)
		}
		lnzMask := pack(lnz, 0)
		unzMask := pack(unz, 0)

		lnz = unpack[d.fc.MBLeft.nzMask>>4]
		unz = unpack[d.fc.MBTop[x].nzMask>>4]
		for c := 0; c < 4; c += 2 {
			for j := range 2 {
				nz := lnz[j+c]
				for i := range 2 {
					nz = d.decodeResid(coefP, PLANE_UorV, nz+unz[i+c], base, quant.uv, mb)
					unz[i+c] = nz
					nzAC[j*2+i] = nz
					nzDC[j*2+i] = btou(mb.Coeffs[base] != 0)
					base += 16
				}
				lnz[j+c] = nz
			}
			nzDCMask |= pack(nzDC, 16+c*2)
			nzACMask |= pack(nzAC, 16+c*2)
		}
		lnzMask |= pack(lnz, 4)
		unzMask |= pack(unz, 4)

		d.fc.MBLeft.nzMask = uint8(lnzMask)
		d.fc.MBTop[x].nzMask = uint8(unzMask)
		d.nzDCMask = nzDCMask
		d.nzACMask = nzACMask

		mb.SkipCoef = nzDCMask == 0 && nzACMask == 0
	} else {
		if mb.UseLuma16 {
			d.fc.MBLeft.nzY16 = 0
			d.fc.MBTop[x].nzY16 = 0
		}

		d.fc.MBLeft.nzMask = 0
		d.fc.MBTop[x].nzMask = 0
		d.nzDCMask = 0
		d.nzACMask = 0
	}

	d.reconstructMB(x, y, mb)

	d.fc.Macroblocks = append(d.fc.Macroblocks, mb)
	return mb
}

func (d *decoder) decodeResid(decoder *compression.Decoder, plane, context uint8, base int, quant [2]uint16, mb *MacroBlock) uint8 {
	prob, n := d.fc.CoeffProbs[plane], 0
	if plane == 0 {
		n = 1
	}

	p := prob[coeffBands[n]][context]
	if !decoder.ReadFlagProb(p[0]) {
		return 0
	}

	for n != 16 {
		n++
		if !decoder.ReadFlagProb(p[1]) {
			p = prob[coeffBands[n]][0]
			continue
		}

		var v uint32
		if !decoder.ReadFlagProb(p[2]) {
			v = 1
			p = prob[coeffBands[n]][1]
		} else {
			if !decoder.ReadFlagProb(p[3]) {
				if !decoder.ReadFlagProb(p[4]) {
					v = 2
				} else {
					v = 3 + decoder.ReadProb(1, p[5])
				}
			} else if !decoder.ReadFlagProb(p[6]) {
				if !decoder.ReadFlagProb(p[7]) {
					v = 5 + decoder.ReadProb(1, 159)
				} else {
					v = 7 + 2*decoder.ReadProb(1, 165) + decoder.ReadProb(1, 145)
				}
			} else {
				b1 := decoder.ReadProb(1, p[8])
				b0 := decoder.ReadProb(1, p[9+b1])
				cat := 2*b1 + b0
				tab := cat3456[cat]
				v = 0
				for i := 0; tab[i] != 0; i++ {
					v *= 2
					v += decoder.ReadProb(1, tab[i])
				}
				v += 3 + (8 << cat)
			}
			p = prob[coeffBands[n]][2]
		}

		z := zigzag[n-1]
		c := int32(v) * int32(quant[btou(z > 0)])
		if decoder.ReadFlag() {
			c = -c
		}
		mb.Coeffs[base+int(z)] = int16(c)
		if n == 16 || !decoder.ReadFlagProb(p[0]) {
			return 1
		}
	}
	return 1
}

func (d *decoder) reconstructMB(mbx, mby int, mb *MacroBlock) {
	if mb.UseLuma16 {
		p := checkTopLeftPred(mbx, mby, uint8(mb.Luma16))
		predFunc16[p](d, 1, 8)
		for j := 0; j < 4; j++ {
			for i := 0; i < 4; i++ {
				n := 4*j + i
				y := 4*j + 1
				x := 4*i + 8
				mask := uint32(1) << uint(n)
				if d.nzACMask&mask != 0 {
					d.inverseDCT4(y, x, 16*n, mb)
				} else if d.nzDCMask&mask != 0 {
					d.inverseDCT4DCOnly(y, x, 16*n, mb)
				}
			}
		}
	} else {
		for j := 0; j < 4; j++ {
			for i := 0; i < 4; i++ {
				n := 4*j + i
				y := 4*j + 1
				x := 4*i + 8
				predFunc4[mb.LumaBModes[j][i]](d, y, x)
				mask := uint32(1) << uint(n)
				if d.nzACMask&mask != 0 {
					d.inverseDCT4(y, x, 16*n, mb)
				} else if d.nzDCMask&mask != 0 {
					d.inverseDCT4DCOnly(y, x, 16*n, mb)
				}
			}
		}
	}

	p := checkTopLeftPred(mbx, mby, uint8(mb.Chroma))

	predFunc8[p](d, 18, 8)
	if d.nzACMask&0x0f0000 != 0 {
		d.inverseDCT8(18, 8, 256, mb)
	} else if d.nzDCMask&0x0f0000 != 0 {
		d.inverseDCT8DCOnly(18, 8, 256, mb)
	}

	predFunc8[p](d, 18, 24)
	if d.nzACMask&0xf00000 != 0 {
		d.inverseDCT8(18, 24, 320, mb)
	} else if d.nzDCMask&0xf00000 != 0 {
		d.inverseDCT8DCOnly(18, 24, 320, mb)
	}

	for i, j := (mby*d.img.YStride+mbx)*16, 0; j < 16; i, j = i+d.img.YStride, j+1 {
		copy(d.img.Y[i:i+16], d.fc.YBR[1+j][8:24])
	}
	for i, j := (mby*d.img.CStride+mbx)*8, 0; j < 8; i, j = i+d.img.CStride, j+1 {
		copy(d.img.Cb[i:i+8], d.fc.YBR[18+j][8:16])
		copy(d.img.Cr[i:i+8], d.fc.YBR[18+j][24:32])
	}
}

func (d *decoder) inverseWHT16(mb *MacroBlock) {
	var m [16]int32
	out := 0

	for i := range 4 {
		a1 := int32(mb.Coeffs[384+0+i]) + int32(mb.Coeffs[384+12+i])
		b1 := int32(mb.Coeffs[384+4+i]) + int32(mb.Coeffs[384+8+i])
		c1 := int32(mb.Coeffs[384+4+i]) - int32(mb.Coeffs[384+8+i])
		d1 := int32(mb.Coeffs[384+0+i]) - int32(mb.Coeffs[384+12+i])
		m[i] = a1 + b1
		m[i+8] = a1 - b1
		m[i+4] = d1 + c1
		m[i+12] = d1 - c1
	}

	for i := range 4 {
		dc := m[i*4] + 3

		a1 := dc + m[3+i*4]
		b1 := m[1+i*4] + m[2+i*4]
		c1 := m[1+i*4] - m[2+i*4]
		d1 := dc - m[3+i*4]

		mb.Coeffs[out] = int16((a1 + b1) >> 3)
		mb.Coeffs[out+16] = int16((d1 + c1) >> 3)
		mb.Coeffs[out+32] = int16((a1 - b1) >> 3)
		mb.Coeffs[out+48] = int16((d1 - c1) >> 3)

		out += 64
	}
}

func (z *decoder) inverseDCT4(y, x, coeffBase int, mb *MacroBlock) {
	const (
		c1 = 85627 // 65536 * cos(pi/8) * sqrt(2).
		c2 = 35468 // 65536 * sin(pi/8) * sqrt(2).
	)
	var m [4][4]int32
	for i := 0; i < 4; i++ {
		a := int32(mb.Coeffs[coeffBase+0]) + int32(mb.Coeffs[coeffBase+8])
		b := int32(mb.Coeffs[coeffBase+0]) - int32(mb.Coeffs[coeffBase+8])
		c := (int32(mb.Coeffs[coeffBase+4])*c2)>>16 - (int32(mb.Coeffs[coeffBase+12])*c1)>>16
		d := (int32(mb.Coeffs[coeffBase+4])*c1)>>16 + (int32(mb.Coeffs[coeffBase+12])*c2)>>16
		m[i][0] = a + d
		m[i][1] = b + c
		m[i][2] = b - c
		m[i][3] = a - d
		coeffBase++
	}
	for j := 0; j < 4; j++ {
		dc := m[0][j] + 4
		a := dc + m[2][j]
		b := dc - m[2][j]
		c := (m[1][j]*c2)>>16 - (m[3][j]*c1)>>16
		d := (m[1][j]*c1)>>16 + (m[3][j]*c2)>>16
		z.fc.YBR[y+j][x+0] = clip8(int32(z.fc.YBR[y+j][x+0]) + (a+d)>>3)
		z.fc.YBR[y+j][x+1] = clip8(int32(z.fc.YBR[y+j][x+1]) + (b+c)>>3)
		z.fc.YBR[y+j][x+2] = clip8(int32(z.fc.YBR[y+j][x+2]) + (b-c)>>3)
		z.fc.YBR[y+j][x+3] = clip8(int32(z.fc.YBR[y+j][x+3]) + (a-d)>>3)
	}
}

func (z *decoder) inverseDCT4DCOnly(y, x, coeffBase int, mb *MacroBlock) {
	dc := (int32(mb.Coeffs[coeffBase+0]) + 4) >> 3
	for j := 0; j < 4; j++ {
		for i := 0; i < 4; i++ {
			z.fc.YBR[y+j][x+i] = clip8(int32(z.fc.YBR[y+j][x+i]) + dc)
		}
	}
}

func (z *decoder) inverseDCT8(y, x, coeffBase int, mb *MacroBlock) {
	z.inverseDCT4(y+0, x+0, coeffBase+0*16, mb)
	z.inverseDCT4(y+0, x+4, coeffBase+1*16, mb)
	z.inverseDCT4(y+4, x+0, coeffBase+2*16, mb)
	z.inverseDCT4(y+4, x+4, coeffBase+3*16, mb)
}

func (z *decoder) inverseDCT8DCOnly(y, x, coeffBase int, mb *MacroBlock) {
	z.inverseDCT4DCOnly(y+0, x+0, coeffBase+0*16, mb)
	z.inverseDCT4DCOnly(y+0, x+4, coeffBase+1*16, mb)
	z.inverseDCT4DCOnly(y+4, x+0, coeffBase+2*16, mb)
	z.inverseDCT4DCOnly(y+4, x+4, coeffBase+3*16, mb)
}

func clip8(i int32) uint8 {
	if i < 0 {
		return 0
	}
	if i > 255 {
		return 255
	}
	return uint8(i)
}

func (d *decoder) prepareYBR(x, y int) {
	if x == 0 {
		for i := range 17 {
			d.fc.YBR[i][7] = 0x81
		}
		for i := range 9 {
			d.fc.YBR[17+i][7] = 0x81
			d.fc.YBR[17+i][23] = 0x81
		}
	} else {
		for i := range 17 {
			d.fc.YBR[i][7] = d.fc.YBR[i][23]
		}
		for i := range 9 {
			d.fc.YBR[17+i][7] = d.fc.YBR[17+i][15]
			d.fc.YBR[17+i][23] = d.fc.YBR[17+i][31]
		}
	}

	if y == 0 {
		for i := range 21 {
			d.fc.YBR[0][7+i] = 0x7f
		}
		for i := range 9 {
			d.fc.YBR[17][7+i] = 0x7f
		}
		for i := range 9 {
			d.fc.YBR[17][23+i] = 0x7f
		}
	} else {
		for i := range 16 {
			d.fc.YBR[0][8+i] = d.img.Y[(16*y-1)*d.img.YStride+16*x+i]
		}
		for i := range 8 {
			d.fc.YBR[17][8+i] = d.img.Cb[(8*y-1)*d.img.CStride+8*x+i]
		}
		for i := range 8 {
			d.fc.YBR[17][24+i] = d.img.Cr[(8*y-1)*d.img.CStride+8*x+i]
		}
		if x == int(d.vp8.MBWidth)-1 {
			for i := range 4 {
				d.fc.YBR[0][24+i] = d.img.Y[(16*y-1)*d.img.YStride+16*x+15]
			}
		} else {
			for i := range 4 {
				d.fc.YBR[0][24+i] = d.img.Y[(16*y-1)*d.img.YStride+16*x+(16+i)]
			}
		}
	}

	for i := 4; i < 16; i += 4 {
		d.fc.YBR[i][24] = d.fc.YBR[0][24]
		d.fc.YBR[i][25] = d.fc.YBR[0][25]
		d.fc.YBR[i][26] = d.fc.YBR[0][26]
		d.fc.YBR[i][27] = d.fc.YBR[0][27]
	}
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
	-predDC, 2,
	-predTM, 4,
	-predVE, 6,
	8, 12,
	-predHE, 10,
	-predRD, -predVR,
	-predLD, 14,
	-predVL, 16,
	-predHD, -predHU,
}

var kfBmodeProb = [10][10][9]uint8{
	{
		{231, 120, 48, 89, 115, 113, 120, 152, 112},
		{152, 179, 64, 126, 170, 118, 46, 70, 95},
		{175, 69, 143, 80, 85, 82, 72, 155, 103},
		{56, 58, 10, 171, 218, 189, 17, 13, 152},
		{114, 26, 17, 163, 44, 195, 21, 10, 173},
		{121, 24, 80, 195, 26, 62, 44, 64, 85},
		{144, 71, 10, 38, 171, 213, 144, 34, 26},
		{170, 46, 55, 19, 136, 160, 33, 206, 71},
		{63, 20, 8, 114, 114, 208, 12, 9, 226},
		{81, 40, 11, 96, 182, 84, 29, 16, 36},
	},
	{
		{134, 183, 89, 137, 98, 101, 106, 165, 148},
		{72, 187, 100, 130, 157, 111, 32, 75, 80},
		{66, 102, 167, 99, 74, 62, 40, 234, 128},
		{41, 53, 9, 178, 241, 141, 26, 8, 107},
		{74, 43, 26, 146, 73, 166, 49, 23, 157},
		{65, 38, 105, 160, 51, 52, 31, 115, 128},
		{104, 79, 12, 27, 217, 255, 87, 17, 7},
		{87, 68, 71, 44, 114, 51, 15, 186, 23},
		{47, 41, 14, 110, 182, 183, 21, 17, 194},
		{66, 45, 25, 102, 197, 189, 23, 18, 22},
	},
	{
		{88, 88, 147, 150, 42, 46, 45, 196, 205},
		{43, 97, 183, 117, 85, 38, 35, 179, 61},
		{39, 53, 200, 87, 26, 21, 43, 232, 171},
		{56, 34, 51, 104, 114, 102, 29, 93, 77},
		{39, 28, 85, 171, 58, 165, 90, 98, 64},
		{34, 22, 116, 206, 23, 34, 43, 166, 73},
		{107, 54, 32, 26, 51, 1, 81, 43, 31},
		{68, 25, 106, 22, 64, 171, 36, 225, 114},
		{34, 19, 21, 102, 132, 188, 16, 76, 124},
		{62, 18, 78, 95, 85, 57, 50, 48, 51},
	},
	{
		{193, 101, 35, 159, 215, 111, 89, 46, 111},
		{60, 148, 31, 172, 219, 228, 21, 18, 111},
		{112, 113, 77, 85, 179, 255, 38, 120, 114},
		{40, 42, 1, 196, 245, 209, 10, 25, 109},
		{88, 43, 29, 140, 166, 213, 37, 43, 154},
		{61, 63, 30, 155, 67, 45, 68, 1, 209},
		{100, 80, 8, 43, 154, 1, 51, 26, 71},
		{142, 78, 78, 16, 255, 128, 34, 197, 171},
		{41, 40, 5, 102, 211, 183, 4, 1, 221},
		{51, 50, 17, 168, 209, 192, 23, 25, 82},
	},
	{
		{138, 31, 36, 171, 27, 166, 38, 44, 229},
		{67, 87, 58, 169, 82, 115, 26, 59, 179},
		{63, 59, 90, 180, 59, 166, 93, 73, 154},
		{40, 40, 21, 116, 143, 209, 34, 39, 175},
		{47, 15, 16, 183, 34, 223, 49, 45, 183},
		{46, 17, 33, 183, 6, 98, 15, 32, 183},
		{57, 46, 22, 24, 128, 1, 54, 17, 37},
		{65, 32, 73, 115, 28, 128, 23, 128, 205},
		{40, 3, 9, 115, 51, 192, 18, 6, 223},
		{87, 37, 9, 115, 59, 77, 64, 21, 47},
	},
	{
		{104, 55, 44, 218, 9, 54, 53, 130, 226},
		{64, 90, 70, 205, 40, 41, 23, 26, 57},
		{54, 57, 112, 184, 5, 41, 38, 166, 213},
		{30, 34, 26, 133, 152, 116, 10, 32, 134},
		{39, 19, 53, 221, 26, 114, 32, 73, 255},
		{31, 9, 65, 234, 2, 15, 1, 118, 73},
		{75, 32, 12, 51, 192, 255, 160, 43, 51},
		{88, 31, 35, 67, 102, 85, 55, 186, 85},
		{56, 21, 23, 111, 59, 205, 45, 37, 192},
		{55, 38, 70, 124, 73, 102, 1, 34, 98},
	},
	{
		{125, 98, 42, 88, 104, 85, 117, 175, 82},
		{95, 84, 53, 89, 128, 100, 113, 101, 45},
		{75, 79, 123, 47, 51, 128, 81, 171, 1},
		{57, 17, 5, 71, 102, 57, 53, 41, 49},
		{38, 33, 13, 121, 57, 73, 26, 1, 85},
		{41, 10, 67, 138, 77, 110, 90, 47, 114},
		{115, 21, 2, 10, 102, 255, 166, 23, 6},
		{101, 29, 16, 10, 85, 128, 101, 196, 26},
		{57, 18, 10, 102, 102, 213, 34, 20, 43},
		{117, 20, 15, 36, 163, 128, 68, 1, 26},
	},
	{
		{102, 61, 71, 37, 34, 53, 31, 243, 192},
		{69, 60, 71, 38, 73, 119, 28, 222, 37},
		{68, 45, 128, 34, 1, 47, 11, 245, 171},
		{62, 17, 19, 70, 146, 85, 55, 62, 70},
		{37, 43, 37, 154, 100, 163, 85, 160, 1},
		{63, 9, 92, 136, 28, 64, 32, 201, 85},
		{75, 15, 9, 9, 64, 255, 184, 119, 16},
		{86, 6, 28, 5, 64, 255, 25, 248, 1},
		{56, 8, 17, 132, 137, 255, 55, 116, 128},
		{58, 15, 20, 82, 135, 57, 26, 121, 40},
	},
	{
		{164, 50, 31, 137, 154, 133, 25, 35, 218},
		{51, 103, 44, 131, 131, 123, 31, 6, 158},
		{86, 40, 64, 135, 148, 224, 45, 183, 128},
		{22, 26, 17, 131, 240, 154, 14, 1, 209},
		{45, 16, 21, 91, 64, 222, 7, 1, 197},
		{56, 21, 39, 155, 60, 138, 23, 102, 213},
		{83, 12, 13, 54, 192, 255, 68, 47, 28},
		{85, 26, 85, 85, 128, 128, 32, 146, 171},
		{18, 11, 7, 63, 144, 171, 4, 4, 246},
		{35, 27, 10, 146, 174, 171, 12, 26, 128},
	},
	{
		{190, 80, 35, 99, 180, 80, 126, 54, 45},
		{85, 126, 47, 87, 176, 51, 41, 20, 32},
		{101, 75, 128, 139, 118, 146, 116, 128, 85},
		{56, 41, 15, 176, 236, 85, 37, 9, 62},
		{71, 30, 17, 119, 118, 255, 17, 18, 138},
		{101, 38, 60, 138, 55, 70, 43, 26, 142},
		{146, 36, 19, 30, 171, 255, 97, 27, 20},
		{138, 45, 61, 62, 219, 1, 81, 188, 64},
		{32, 41, 20, 117, 151, 142, 20, 21, 163},
		{112, 19, 12, 61, 195, 128, 48, 4, 24},
	},
}

var uvModeTree = [6]int8{
	-predDC, 2,
	-predVE, 4,
	-predHE, -predTM,
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

var Pcat1 = [2]uint8{159, 0}
var Pcat2 = [3]uint8{165, 145, 0}
var Pcat3 = [4]uint8{173, 148, 140, 0}
var Pcat4 = [5]uint8{176, 155, 140, 135, 0}
var Pcat5 = [6]uint8{180, 157, 141, 134, 130, 0}
var Pcat6 = [12]uint8{254, 254, 243, 230, 196, 177, 153, 140, 133, 130, 129, 0}

var cat3456 = [4][12]uint8{
	{173, 148, 140, 0, 0, 0, 0, 0, 0, 0, 0, 0},
	{176, 155, 140, 135, 0, 0, 0, 0, 0, 0, 0, 0},
	{180, 157, 141, 134, 130, 0, 0, 0, 0, 0, 0, 0},
	{254, 254, 243, 230, 196, 177, 153, 140, 133, 130, 129, 0},
}

var coeffBands = [17]int{0, 1, 2, 3, 6, 4, 5, 6, 6, 6, 6, 6, 6, 6, 6, 7, 0}

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

func btou(b bool) uint8 {
	if b {
		return 1
	}
	return 0
}

func pack(x [4]uint8, shift int) uint32 {
	u := uint32(x[0])<<0 | uint32(x[1])<<1 | uint32(x[2])<<2 | uint32(x[3])<<3
	return u << uint(shift)
}

func clip(x, min, max int32) int32 {
	if x < min {
		return min
	}
	if x > max {
		return max
	}
	return x
}

var (
	dequantTableDC = [128]uint16{
		4, 5, 6, 7, 8, 9, 10, 10,
		11, 12, 13, 14, 15, 16, 17, 17,
		18, 19, 20, 20, 21, 21, 22, 22,
		23, 23, 24, 25, 25, 26, 27, 28,
		29, 30, 31, 32, 33, 34, 35, 36,
		37, 37, 38, 39, 40, 41, 42, 43,
		44, 45, 46, 46, 47, 48, 49, 50,
		51, 52, 53, 54, 55, 56, 57, 58,
		59, 60, 61, 62, 63, 64, 65, 66,
		67, 68, 69, 70, 71, 72, 73, 74,
		75, 76, 76, 77, 78, 79, 80, 81,
		82, 83, 84, 85, 86, 87, 88, 89,
		91, 93, 95, 96, 98, 100, 101, 102,
		104, 106, 108, 110, 112, 114, 116, 118,
		122, 124, 126, 128, 130, 132, 134, 136,
		138, 140, 143, 145, 148, 151, 154, 157,
	}
	dequantTableAC = [128]uint16{
		4, 5, 6, 7, 8, 9, 10, 11,
		12, 13, 14, 15, 16, 17, 18, 19,
		20, 21, 22, 23, 24, 25, 26, 27,
		28, 29, 30, 31, 32, 33, 34, 35,
		36, 37, 38, 39, 40, 41, 42, 43,
		44, 45, 46, 47, 48, 49, 50, 51,
		52, 53, 54, 55, 56, 57, 58, 60,
		62, 64, 66, 68, 70, 72, 74, 76,
		78, 80, 82, 84, 86, 88, 90, 92,
		94, 96, 98, 100, 102, 104, 106, 108,
		110, 112, 114, 116, 119, 122, 125, 128,
		131, 134, 137, 140, 143, 146, 149, 152,
		155, 158, 161, 164, 167, 170, 173, 177,
		181, 185, 189, 193, 197, 201, 205, 209,
		213, 217, 221, 225, 229, 234, 239, 245,
		249, 254, 259, 264, 269, 274, 279, 284,
	}
)

const (
	predDC = iota
	predTM
	predVE
	predHE
	predRD
	predVR
	predLD
	predVL
	predHD
	predHU
	predDCTop
	predDCLeft
	predDCTopLeft
)

func checkTopLeftPred(mbx, mby int, p uint8) uint8 {
	if p != predDC {
		return p
	}
	if mbx == 0 {
		if mby == 0 {
			return predDCTopLeft
		}
		return predDCLeft
	}
	if mby == 0 {
		return predDCTop
	}
	return predDC
}

var predFunc4 = [...]func(*decoder, int, int){
	predFunc4DC,
	predFunc4TM,
	predFunc4VE,
	predFunc4HE,
	predFunc4RD,
	predFunc4VR,
	predFunc4LD,
	predFunc4VL,
	predFunc4HD,
	predFunc4HU,
	nil,
	nil,
	nil,
}

var predFunc8 = [...]func(*decoder, int, int){
	predFunc8DC,
	predFunc8TM,
	predFunc8VE,
	predFunc8HE,
	nil,
	nil,
	nil,
	nil,
	nil,
	nil,
	predFunc8DCTop,
	predFunc8DCLeft,
	predFunc8DCTopLeft,
}

var predFunc16 = [...]func(*decoder, int, int){
	predFunc16DC,
	predFunc16TM,
	predFunc16VE,
	predFunc16HE,
	nil,
	nil,
	nil,
	nil,
	nil,
	nil,
	predFunc16DCTop,
	predFunc16DCLeft,
	predFunc16DCTopLeft,
}

func predFunc4DC(z *decoder, y, x int) {
	sum := uint32(4)
	for i := 0; i < 4; i++ {
		sum += uint32(z.fc.YBR[y-1][x+i])
	}
	for j := 0; j < 4; j++ {
		sum += uint32(z.fc.YBR[y+j][x-1])
	}
	avg := uint8(sum / 8)
	for j := 0; j < 4; j++ {
		for i := 0; i < 4; i++ {
			z.fc.YBR[y+j][x+i] = avg
		}
	}
}

func predFunc4TM(z *decoder, y, x int) {
	delta0 := -int32(z.fc.YBR[y-1][x-1])
	for j := 0; j < 4; j++ {
		delta1 := delta0 + int32(z.fc.YBR[y+j][x-1])
		for i := 0; i < 4; i++ {
			delta2 := delta1 + int32(z.fc.YBR[y-1][x+i])
			z.fc.YBR[y+j][x+i] = uint8(clip(delta2, 0, 255))
		}
	}
}

func predFunc4VE(z *decoder, y, x int) {
	a := int32(z.fc.YBR[y-1][x-1])
	b := int32(z.fc.YBR[y-1][x+0])
	c := int32(z.fc.YBR[y-1][x+1])
	d := int32(z.fc.YBR[y-1][x+2])
	e := int32(z.fc.YBR[y-1][x+3])
	f := int32(z.fc.YBR[y-1][x+4])
	abc := uint8((a + 2*b + c + 2) / 4)
	bcd := uint8((b + 2*c + d + 2) / 4)
	cde := uint8((c + 2*d + e + 2) / 4)
	def := uint8((d + 2*e + f + 2) / 4)
	for j := 0; j < 4; j++ {
		z.fc.YBR[y+j][x+0] = abc
		z.fc.YBR[y+j][x+1] = bcd
		z.fc.YBR[y+j][x+2] = cde
		z.fc.YBR[y+j][x+3] = def
	}
}

func predFunc4HE(z *decoder, y, x int) {
	s := int32(z.fc.YBR[y+3][x-1])
	r := int32(z.fc.YBR[y+2][x-1])
	q := int32(z.fc.YBR[y+1][x-1])
	p := int32(z.fc.YBR[y+0][x-1])
	a := int32(z.fc.YBR[y-1][x-1])
	ssr := uint8((s + 2*s + r + 2) / 4)
	srq := uint8((s + 2*r + q + 2) / 4)
	rqp := uint8((r + 2*q + p + 2) / 4)
	apq := uint8((a + 2*p + q + 2) / 4)
	for i := 0; i < 4; i++ {
		z.fc.YBR[y+0][x+i] = apq
		z.fc.YBR[y+1][x+i] = rqp
		z.fc.YBR[y+2][x+i] = srq
		z.fc.YBR[y+3][x+i] = ssr
	}
}

func predFunc4RD(z *decoder, y, x int) {
	s := int32(z.fc.YBR[y+3][x-1])
	r := int32(z.fc.YBR[y+2][x-1])
	q := int32(z.fc.YBR[y+1][x-1])
	p := int32(z.fc.YBR[y+0][x-1])
	a := int32(z.fc.YBR[y-1][x-1])
	b := int32(z.fc.YBR[y-1][x+0])
	c := int32(z.fc.YBR[y-1][x+1])
	d := int32(z.fc.YBR[y-1][x+2])
	e := int32(z.fc.YBR[y-1][x+3])
	srq := uint8((s + 2*r + q + 2) / 4)
	rqp := uint8((r + 2*q + p + 2) / 4)
	qpa := uint8((q + 2*p + a + 2) / 4)
	pab := uint8((p + 2*a + b + 2) / 4)
	abc := uint8((a + 2*b + c + 2) / 4)
	bcd := uint8((b + 2*c + d + 2) / 4)
	cde := uint8((c + 2*d + e + 2) / 4)
	z.fc.YBR[y+0][x+0] = pab
	z.fc.YBR[y+0][x+1] = abc
	z.fc.YBR[y+0][x+2] = bcd
	z.fc.YBR[y+0][x+3] = cde
	z.fc.YBR[y+1][x+0] = qpa
	z.fc.YBR[y+1][x+1] = pab
	z.fc.YBR[y+1][x+2] = abc
	z.fc.YBR[y+1][x+3] = bcd
	z.fc.YBR[y+2][x+0] = rqp
	z.fc.YBR[y+2][x+1] = qpa
	z.fc.YBR[y+2][x+2] = pab
	z.fc.YBR[y+2][x+3] = abc
	z.fc.YBR[y+3][x+0] = srq
	z.fc.YBR[y+3][x+1] = rqp
	z.fc.YBR[y+3][x+2] = qpa
	z.fc.YBR[y+3][x+3] = pab
}

func predFunc4VR(z *decoder, y, x int) {
	r := int32(z.fc.YBR[y+2][x-1])
	q := int32(z.fc.YBR[y+1][x-1])
	p := int32(z.fc.YBR[y+0][x-1])
	a := int32(z.fc.YBR[y-1][x-1])
	b := int32(z.fc.YBR[y-1][x+0])
	c := int32(z.fc.YBR[y-1][x+1])
	d := int32(z.fc.YBR[y-1][x+2])
	e := int32(z.fc.YBR[y-1][x+3])
	ab := uint8((a + b + 1) / 2)
	bc := uint8((b + c + 1) / 2)
	cd := uint8((c + d + 1) / 2)
	de := uint8((d + e + 1) / 2)
	rqp := uint8((r + 2*q + p + 2) / 4)
	qpa := uint8((q + 2*p + a + 2) / 4)
	pab := uint8((p + 2*a + b + 2) / 4)
	abc := uint8((a + 2*b + c + 2) / 4)
	bcd := uint8((b + 2*c + d + 2) / 4)
	cde := uint8((c + 2*d + e + 2) / 4)
	z.fc.YBR[y+0][x+0] = ab
	z.fc.YBR[y+0][x+1] = bc
	z.fc.YBR[y+0][x+2] = cd
	z.fc.YBR[y+0][x+3] = de
	z.fc.YBR[y+1][x+0] = pab
	z.fc.YBR[y+1][x+1] = abc
	z.fc.YBR[y+1][x+2] = bcd
	z.fc.YBR[y+1][x+3] = cde
	z.fc.YBR[y+2][x+0] = qpa
	z.fc.YBR[y+2][x+1] = ab
	z.fc.YBR[y+2][x+2] = bc
	z.fc.YBR[y+2][x+3] = cd
	z.fc.YBR[y+3][x+0] = rqp
	z.fc.YBR[y+3][x+1] = pab
	z.fc.YBR[y+3][x+2] = abc
	z.fc.YBR[y+3][x+3] = bcd
}

func predFunc4LD(z *decoder, y, x int) {
	a := int32(z.fc.YBR[y-1][x+0])
	b := int32(z.fc.YBR[y-1][x+1])
	c := int32(z.fc.YBR[y-1][x+2])
	d := int32(z.fc.YBR[y-1][x+3])
	e := int32(z.fc.YBR[y-1][x+4])
	f := int32(z.fc.YBR[y-1][x+5])
	g := int32(z.fc.YBR[y-1][x+6])
	h := int32(z.fc.YBR[y-1][x+7])
	abc := uint8((a + 2*b + c + 2) / 4)
	bcd := uint8((b + 2*c + d + 2) / 4)
	cde := uint8((c + 2*d + e + 2) / 4)
	def := uint8((d + 2*e + f + 2) / 4)
	efg := uint8((e + 2*f + g + 2) / 4)
	fgh := uint8((f + 2*g + h + 2) / 4)
	ghh := uint8((g + 2*h + h + 2) / 4)
	z.fc.YBR[y+0][x+0] = abc
	z.fc.YBR[y+0][x+1] = bcd
	z.fc.YBR[y+0][x+2] = cde
	z.fc.YBR[y+0][x+3] = def
	z.fc.YBR[y+1][x+0] = bcd
	z.fc.YBR[y+1][x+1] = cde
	z.fc.YBR[y+1][x+2] = def
	z.fc.YBR[y+1][x+3] = efg
	z.fc.YBR[y+2][x+0] = cde
	z.fc.YBR[y+2][x+1] = def
	z.fc.YBR[y+2][x+2] = efg
	z.fc.YBR[y+2][x+3] = fgh
	z.fc.YBR[y+3][x+0] = def
	z.fc.YBR[y+3][x+1] = efg
	z.fc.YBR[y+3][x+2] = fgh
	z.fc.YBR[y+3][x+3] = ghh
}

func predFunc4VL(z *decoder, y, x int) {
	a := int32(z.fc.YBR[y-1][x+0])
	b := int32(z.fc.YBR[y-1][x+1])
	c := int32(z.fc.YBR[y-1][x+2])
	d := int32(z.fc.YBR[y-1][x+3])
	e := int32(z.fc.YBR[y-1][x+4])
	f := int32(z.fc.YBR[y-1][x+5])
	g := int32(z.fc.YBR[y-1][x+6])
	h := int32(z.fc.YBR[y-1][x+7])
	ab := uint8((a + b + 1) / 2)
	bc := uint8((b + c + 1) / 2)
	cd := uint8((c + d + 1) / 2)
	de := uint8((d + e + 1) / 2)
	abc := uint8((a + 2*b + c + 2) / 4)
	bcd := uint8((b + 2*c + d + 2) / 4)
	cde := uint8((c + 2*d + e + 2) / 4)
	def := uint8((d + 2*e + f + 2) / 4)
	efg := uint8((e + 2*f + g + 2) / 4)
	fgh := uint8((f + 2*g + h + 2) / 4)
	z.fc.YBR[y+0][x+0] = ab
	z.fc.YBR[y+0][x+1] = bc
	z.fc.YBR[y+0][x+2] = cd
	z.fc.YBR[y+0][x+3] = de
	z.fc.YBR[y+1][x+0] = abc
	z.fc.YBR[y+1][x+1] = bcd
	z.fc.YBR[y+1][x+2] = cde
	z.fc.YBR[y+1][x+3] = def
	z.fc.YBR[y+2][x+0] = bc
	z.fc.YBR[y+2][x+1] = cd
	z.fc.YBR[y+2][x+2] = de
	z.fc.YBR[y+2][x+3] = efg
	z.fc.YBR[y+3][x+0] = bcd
	z.fc.YBR[y+3][x+1] = cde
	z.fc.YBR[y+3][x+2] = def
	z.fc.YBR[y+3][x+3] = fgh
}

func predFunc4HD(z *decoder, y, x int) {
	s := int32(z.fc.YBR[y+3][x-1])
	r := int32(z.fc.YBR[y+2][x-1])
	q := int32(z.fc.YBR[y+1][x-1])
	p := int32(z.fc.YBR[y+0][x-1])
	a := int32(z.fc.YBR[y-1][x-1])
	b := int32(z.fc.YBR[y-1][x+0])
	c := int32(z.fc.YBR[y-1][x+1])
	d := int32(z.fc.YBR[y-1][x+2])
	sr := uint8((s + r + 1) / 2)
	rq := uint8((r + q + 1) / 2)
	qp := uint8((q + p + 1) / 2)
	pa := uint8((p + a + 1) / 2)
	srq := uint8((s + 2*r + q + 2) / 4)
	rqp := uint8((r + 2*q + p + 2) / 4)
	qpa := uint8((q + 2*p + a + 2) / 4)
	pab := uint8((p + 2*a + b + 2) / 4)
	abc := uint8((a + 2*b + c + 2) / 4)
	bcd := uint8((b + 2*c + d + 2) / 4)
	z.fc.YBR[y+0][x+0] = pa
	z.fc.YBR[y+0][x+1] = pab
	z.fc.YBR[y+0][x+2] = abc
	z.fc.YBR[y+0][x+3] = bcd
	z.fc.YBR[y+1][x+0] = qp
	z.fc.YBR[y+1][x+1] = qpa
	z.fc.YBR[y+1][x+2] = pa
	z.fc.YBR[y+1][x+3] = pab
	z.fc.YBR[y+2][x+0] = rq
	z.fc.YBR[y+2][x+1] = rqp
	z.fc.YBR[y+2][x+2] = qp
	z.fc.YBR[y+2][x+3] = qpa
	z.fc.YBR[y+3][x+0] = sr
	z.fc.YBR[y+3][x+1] = srq
	z.fc.YBR[y+3][x+2] = rq
	z.fc.YBR[y+3][x+3] = rqp
}

func predFunc4HU(z *decoder, y, x int) {
	s := int32(z.fc.YBR[y+3][x-1])
	r := int32(z.fc.YBR[y+2][x-1])
	q := int32(z.fc.YBR[y+1][x-1])
	p := int32(z.fc.YBR[y+0][x-1])
	pq := uint8((p + q + 1) / 2)
	qr := uint8((q + r + 1) / 2)
	rs := uint8((r + s + 1) / 2)
	pqr := uint8((p + 2*q + r + 2) / 4)
	qrs := uint8((q + 2*r + s + 2) / 4)
	rss := uint8((r + 2*s + s + 2) / 4)
	sss := uint8(s)
	z.fc.YBR[y+0][x+0] = pq
	z.fc.YBR[y+0][x+1] = pqr
	z.fc.YBR[y+0][x+2] = qr
	z.fc.YBR[y+0][x+3] = qrs
	z.fc.YBR[y+1][x+0] = qr
	z.fc.YBR[y+1][x+1] = qrs
	z.fc.YBR[y+1][x+2] = rs
	z.fc.YBR[y+1][x+3] = rss
	z.fc.YBR[y+2][x+0] = rs
	z.fc.YBR[y+2][x+1] = rss
	z.fc.YBR[y+2][x+2] = sss
	z.fc.YBR[y+2][x+3] = sss
	z.fc.YBR[y+3][x+0] = sss
	z.fc.YBR[y+3][x+1] = sss
	z.fc.YBR[y+3][x+2] = sss
	z.fc.YBR[y+3][x+3] = sss
}

func predFunc8DC(z *decoder, y, x int) {
	sum := uint32(8)
	for i := 0; i < 8; i++ {
		sum += uint32(z.fc.YBR[y-1][x+i])
	}
	for j := 0; j < 8; j++ {
		sum += uint32(z.fc.YBR[y+j][x-1])
	}
	avg := uint8(sum / 16)
	for j := 0; j < 8; j++ {
		for i := 0; i < 8; i++ {
			z.fc.YBR[y+j][x+i] = avg
		}
	}
}

func predFunc8TM(z *decoder, y, x int) {
	delta0 := -int32(z.fc.YBR[y-1][x-1])
	for j := 0; j < 8; j++ {
		delta1 := delta0 + int32(z.fc.YBR[y+j][x-1])
		for i := 0; i < 8; i++ {
			delta2 := delta1 + int32(z.fc.YBR[y-1][x+i])
			z.fc.YBR[y+j][x+i] = uint8(clip(delta2, 0, 255))
		}
	}
}

func predFunc8VE(z *decoder, y, x int) {
	for j := 0; j < 8; j++ {
		for i := 0; i < 8; i++ {
			z.fc.YBR[y+j][x+i] = z.fc.YBR[y-1][x+i]
		}
	}
}

func predFunc8HE(z *decoder, y, x int) {
	for j := 0; j < 8; j++ {
		for i := 0; i < 8; i++ {
			z.fc.YBR[y+j][x+i] = z.fc.YBR[y+j][x-1]
		}
	}
}

func predFunc8DCTop(z *decoder, y, x int) {
	sum := uint32(4)
	for j := 0; j < 8; j++ {
		sum += uint32(z.fc.YBR[y+j][x-1])
	}
	avg := uint8(sum / 8)
	for j := 0; j < 8; j++ {
		for i := 0; i < 8; i++ {
			z.fc.YBR[y+j][x+i] = avg
		}
	}
}

func predFunc8DCLeft(z *decoder, y, x int) {
	sum := uint32(4)
	for i := 0; i < 8; i++ {
		sum += uint32(z.fc.YBR[y-1][x+i])
	}
	avg := uint8(sum / 8)
	for j := 0; j < 8; j++ {
		for i := 0; i < 8; i++ {
			z.fc.YBR[y+j][x+i] = avg
		}
	}
}

func predFunc8DCTopLeft(z *decoder, y, x int) {
	for j := 0; j < 8; j++ {
		for i := 0; i < 8; i++ {
			z.fc.YBR[y+j][x+i] = 0x80
		}
	}
}

func predFunc16DC(z *decoder, y, x int) {
	sum := uint32(16)
	for i := 0; i < 16; i++ {
		sum += uint32(z.fc.YBR[y-1][x+i])
	}
	for j := 0; j < 16; j++ {
		sum += uint32(z.fc.YBR[y+j][x-1])
	}
	avg := uint8(sum / 32)
	for j := 0; j < 16; j++ {
		for i := 0; i < 16; i++ {
			z.fc.YBR[y+j][x+i] = avg
		}
	}
}

func predFunc16TM(z *decoder, y, x int) {
	delta0 := -int32(z.fc.YBR[y-1][x-1])
	for j := 0; j < 16; j++ {
		delta1 := delta0 + int32(z.fc.YBR[y+j][x-1])
		for i := 0; i < 16; i++ {
			delta2 := delta1 + int32(z.fc.YBR[y-1][x+i])
			z.fc.YBR[y+j][x+i] = uint8(clip(delta2, 0, 255))
		}
	}
}

func predFunc16VE(z *decoder, y, x int) {
	for j := 0; j < 16; j++ {
		for i := 0; i < 16; i++ {
			z.fc.YBR[y+j][x+i] = z.fc.YBR[y-1][x+i]
		}
	}
}

func predFunc16HE(z *decoder, y, x int) {
	for j := 0; j < 16; j++ {
		for i := 0; i < 16; i++ {
			z.fc.YBR[y+j][x+i] = z.fc.YBR[y+j][x-1]
		}
	}
}

func predFunc16DCTop(z *decoder, y, x int) {
	sum := uint32(8)
	for j := 0; j < 16; j++ {
		sum += uint32(z.fc.YBR[y+j][x-1])
	}
	avg := uint8(sum / 16)
	for j := 0; j < 16; j++ {
		for i := 0; i < 16; i++ {
			z.fc.YBR[y+j][x+i] = avg
		}
	}
}

func predFunc16DCLeft(z *decoder, y, x int) {
	sum := uint32(8)
	for i := 0; i < 16; i++ {
		sum += uint32(z.fc.YBR[y-1][x+i])
	}
	avg := uint8(sum / 16)
	for j := 0; j < 16; j++ {
		for i := 0; i < 16; i++ {
			z.fc.YBR[y+j][x+i] = avg
		}
	}
}

func predFunc16DCTopLeft(z *decoder, y, x int) {
	for j := 0; j < 16; j++ {
		for i := 0; i < 16; i++ {
			z.fc.YBR[y+j][x+i] = 0x80
		}
	}
}
