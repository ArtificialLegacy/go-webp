package compression

type Encoder struct {
	Output []byte

	Bottom   uint32
	Range    uint32
	BitCount int
}

func NewEncoder() *Encoder {
	return &Encoder{
		Output: []byte{},

		Range:    255,
		BitCount: 24,
	}
}

func (e *Encoder) Write(b byte, p uint8) {
	for i := range 8 {
		bv := ((b >> i) & 1) == 1

		split := 1 + (((e.Range - 1) * uint32(p)) >> 8)

		if bv {
			e.Bottom += split
			e.Range -= split
		} else {
			e.Range = split
		}

		for e.Range < 128 {
			e.Range <<= 1

			if e.Bottom&(1<<31) != 0 {
				e.carry()
			}

			e.Bottom <<= 1
			e.BitCount--
			if e.BitCount == 0 {
				e.Output = append(e.Output, uint8(e.Bottom>>24))
				e.Bottom &= (1 << 24) - 1
				e.BitCount = 8
			}
		}
	}
}

func (e *Encoder) Flush() {
	c := e.BitCount
	v := e.Bottom

	if v&(1<<(32-uint32(c))) != 0 {
		e.carry()
	}

	v <<= c & 7
	c >>= 3

	for ; c > 0; c-- {
		v <<= 8
	}
	c = 4

	for ; c > 0; c-- {
		e.Output = append(e.Output, uint8(v>>24))
		v <<= 8
	}
}

func (e *Encoder) carry() {
	pos := len(e.Output) - 1

	for ; e.Output[pos] == 255; pos-- {
		e.Output[pos] = 0
	}

	e.Output[pos]++
}
