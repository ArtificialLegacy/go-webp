package compression

type Decoder struct {
	Input []byte
	pos   int

	Range    uint32
	Value    uint32
	BitCount int
}

func NewDecoder(input []byte) *Decoder {
	if len(input) < 2 {
		return nil
	}

	return &Decoder{
		Input: input,
		Range: 255,
		pos:   2,
		Value: (uint32(input[0]) << 8) | uint32(input[1]),
	}
}

func (d *Decoder) Read(bitCount int, prob uint8) uint32 {
	v := uint32(0)
	bc := bitCount - 1

	for ; bitCount > 0; bitCount-- {
		v = (v >> 1) + (d.read(prob) << uint32(bc))
	}

	return v
}

func (d *Decoder) read(p uint8) uint32 {
	split := 1 + (((d.Range - 1) * uint32(p)) >> 8)
	SPLIT := split << 8
	var retval uint32

	if d.Value >= SPLIT {
		retval = 1

		d.Range -= split
		d.Value -= SPLIT
	} else {
		retval = 0
		d.Range = split
	}

	for d.Range < 128 {
		d.Value <<= 1
		d.Range <<= 1

		d.BitCount++
		if d.BitCount == 8 {
			d.BitCount = 0
			d.Value |= uint32(d.Input[d.pos])
			d.pos++
		}
	}

	return retval
}
