; Golden-evaluation excerpt regenerated from memdump_dir/landed_blood.bin.
; The older broad listing is from a different runtime image at this range.

1D1B  86 A1       STX $A1
1D1D  A6 A0       LDX $A0
1D1F  D0 02       BNE $1D23
1D21  E6 A0       INC $A0
1D23  BD 00 04    LDA $0400,X
1D26  65 A0       ADC $A0
1D28  85 A0       STA $A0
1D2A  9D 00 04    STA $0400,X
1D2D  A6 A1       LDX $A1
1D2F  60          RTS

1D30  20 1B 1D    JSR $1D1B
1D33  48          PHA
1D34  20 1B 1D    JSR $1D1B
1D37  AA          TAX
1D38  68          PLA
1D39  60          RTS

1D3A  85 BE       STA $BE
1D3C  86 BF       STX $BF
1D3E  A2 00       LDX #$00
1D40  20 1B 1D    JSR $1D1B
1D43  25 BF       AND $BF
1D45  C5 BE       CMP $BE
1D47  90 06       BCC $1D4F
1D49  CA          DEX
1D4A  D0 F4       BNE $1D40
1D4C  38          SEC
1D4D  E5 BE       SBC $BE
1D4F  A6 BF       LDX $BF
1D51  60          RTS
