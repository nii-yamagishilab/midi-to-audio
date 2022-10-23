import sys

"""
This script generate utt2spk from the input file (wav.scp/segments)
The speaker is "piano" on maestro dataset.
In the multi-instrument generation, the speaker could be different instruments.
"""

def main(src_file, utt2spk):
    f_src = open(src_file, 'r')
    lines_src = f_src.readlines()

    f_utt2spk = open(utt2spk, 'w')

    for item in lines_src:
        utt_id = item.strip().split()[0]
        f_utt2spk.write("{} {}\n".format(utt_id, 'piano'))
    
    f_utt2spk.close()


if __name__ == '__main__':
    src = sys.argv[1]
    utt2spk = sys.argv[2]
    main(src, utt2spk)