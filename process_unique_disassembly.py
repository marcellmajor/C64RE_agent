import sys
import re

def process_unique_disassembly(input_file, output_file):
    unique_instructions = {}
    byte_coverage = {}
    
    try:
        with open(input_file, 'r') as fin:
            current_labels = []
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                # Only process lines starting with .C:
                if not line.startswith('.C:'):
                    continue
                # Example line:
                # .C:a565  C9 0D       CMP #$0D       - A:52 X:00 Y:01 SP:f8 ..-.....  155280146
                # Split by whitespace
                parts = line.split()
                if len(parts) < 4:
                    continue
                # Address is after .C:
                addr_str = parts[0][3:]
                try:
                    addr = int(addr_str, 16)
                except ValueError:
                    continue
                # Bytes: all hex pairs after address, before mnemonic (find first non-hex word)
                bytes_list = []
                mnemonic_idx = None
                for i, p in enumerate(parts[1:], 1):
                    if re.fullmatch(r'[A-Za-z]{3}', p):
                        mnemonic_idx = i
                        break
                    if re.fullmatch(r'[0-9A-Fa-f]{2}', p):
                        bytes_list.append(p)
                if mnemonic_idx is None:
                    continue
                # Remove any trailing fields starting with '-' (e.g., '- A:52 X:00 ...')
                inst_fields = []
                for p in parts[mnemonic_idx:]:
                    if p.startswith('-'):
                        break
                    inst_fields.append(p)
                inst = ' '.join(inst_fields)
                bytes_str_clean = ' '.join(bytes_list)
                num_bytes = len(bytes_list)
                if num_bytes == 0:
                    num_bytes = 1 # Fallback
                # If we've already registered this address, skip
                if addr in unique_instructions:
                    continue
                unique_instructions[addr] = {
                    "labels": [],
                    "bytes": bytes_str_clean,
                    "inst": inst,
                    "num_bytes": num_bytes
                }
                for i in range(num_bytes):
                    target = addr + i
                    byte_coverage.setdefault(target, set()).add(addr)
        # Write output sorted by address
        with open(output_file, 'w') as fout:
            for addr in sorted(unique_instructions.keys()):
                info = unique_instructions[addr]
                for lbl in info["labels"]:
                    fout.write(lbl + "\n")
                bytes_formatted = info["bytes"].ljust(11)
                inst_formatted = info["inst"].ljust(15)
                out_line = f"{addr:04x}  {bytes_formatted} {inst_formatted}"
                comments = []
                # This address is embedded inside other instruction(s)
                embedded_in = [b for b in byte_coverage.get(addr, set()) if b != addr]
                if embedded_in:
                    bases = [f"{b:04x}" for b in sorted(embedded_in)]
                    comments.append(f"Polymorphic code overlap: this address is within the bytes of instruction(s) at {', '.join(bases)}")
                # This instruction contains other instruction start(s) inside its bytes
                contains_starts = []
                for i in range(1, info["num_bytes"]):
                    target_addr = addr + i
                    if target_addr in unique_instructions:
                        contains_starts.append(f"{target_addr:04x}")
                if contains_starts:
                    comments.append(f"Polymorphic code overlap: contains instruction(s) starting at {', '.join(contains_starts)}")
                if comments:
                    out_line += " ; " + " | ".join(comments)
                fout.write(out_line + "\n")
        print(f"Successfully wrote {len(unique_instructions)} unique instructions to {output_file}")
    except FileNotFoundError:
        print(f"Error: Could not find {input_file}")

if __name__ == "__main__":
    if len(sys.argv) == 1 or "-h" in sys.argv or "--help" in sys.argv:
        print(f"Usage: python {sys.argv[0]} <input_trace_file> [output_asm_file]")
        print("Extracts unique instructions from a trace file.")
        sys.exit(0)

    input_file = sys.argv[1]
    output_file = "trace_unique_reduced.asm"
    
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
        
    process_unique_disassembly(input_file, output_file)
