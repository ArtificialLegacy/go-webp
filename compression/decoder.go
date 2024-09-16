package compression

type Decoder struct {
	Input []byte
	Pos   int

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
		Pos:   2,
		Value: (uint32(input[0]) << 8) | uint32(input[1]),
	}
}

func (d *Decoder) ReadProb(bitCount int, prob uint8) uint32 {
	v := uint32(0)
	/*bc := bitCount - 1

	for ; bitCount > 0; bitCount-- {
		v = (v >> 1) + (d.read(prob) << uint32(bc))
	}*/

	for ; bitCount > 0; bitCount-- {
		v = (v << 1) + d.read(prob)
	}

	return v
}

func (d *Decoder) Read(bitCount int) uint32 {
	return d.ReadProb(bitCount, 128)
}

func (d *Decoder) Read8(bitCount int) uint8 {
	return uint8(d.ReadProb(bitCount, 128))
}

func (d *Decoder) Read8Prob(bitCount int, prob uint8) uint8 {
	return uint8(d.ReadProb(bitCount, prob))
}

func (d *Decoder) ReadFlag() bool {
	return d.ReadProb(1, 128) == 1
}

func (d *Decoder) ReadFlagProb(prob uint8) bool {
	return d.ReadProb(1, prob) == 1
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
			d.Value |= uint32(d.Input[d.Pos])
			d.Pos++
		}
	}

	return retval
}
