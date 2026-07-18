; Golden-evaluation excerpt regenerated from memdump_dir/petch_ingame.bin.
; It fills the two ranges absent from the broader trace-derived listing.

1F6C  9D 25 0D    STA $0D25,X
1F6F  4C 85 19    JMP $1985
1F72  A9 04       LDA #$04
1F74  4C 81 5A    JMP $5A81
1F77  AD 01 D0    LDA $D001
1F7A  29 0F       AND #$0F
1F7C  C9 00       CMP #$00
1F7E  F0 0B       BEQ $1F8B
1F80  C9 02       CMP #$02
1F82  F0 07       BEQ $1F8B

5A61  A9 01       LDA #$01
5A63  8D 01 79    STA $7901
5A66  A9 20       LDA #$20
5A68  8D 02 79    STA $7902
5A6B  A9 01       LDA #$01
5A6D  8D 00 79    STA $7900
5A70  60          RTS
5A71  20 61 5A    JSR $5A61
5A74  A9 01       LDA #$01
5A76  4C 8A 1E    JMP $1E8A
5A79  20 61 5A    JSR $5A61
5A7C  A9 02       LDA #$02
5A7E  4C FB 1E    JMP $1EFB
5A81  20 61 5A    JSR $5A61
5A84  A9 04       LDA #$04
5A86  4C 6C 1F    JMP $1F6C
5A89  20 61 5A    JSR $5A61
5A8C  A9 03       LDA #$03
5A8E  4C DD 1F    JMP $1FDD
5A91  9D 25 0D    STA $0D25,X
5A94  AD 25 0D    LDA $0D25
5A97  D0 0D       BNE $5AA6
