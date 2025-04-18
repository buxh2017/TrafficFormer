
import os
import sys

this_file_path = os.path.abspath(__file__)
ROOT_DIR = os.path.dirname(os.path.dirname(this_file_path))
sys.path.append(ROOT_DIR)
DATA_DIR = os.path.join(ROOT_DIR, "data")

from data_generation.pretrain_data_gen import corpora_to_bigram, pretrain_dataset_generation
from data_generation.vocab_gen import build_BPE, build_vocab


def pretrain_data():
    input_dir = os.path.join(DATA_DIR, "NonVPN-PCAPs-01")
    output_dir = f"{DATA_DIR}/output"
    pcap_output_dir = f"{output_dir}/pcap"
    os.makedirs(pcap_output_dir, exist_ok=True)
    burst_dir = f"{output_dir}/burst"
    os.makedirs(burst_dir, exist_ok=True)
    pretrain_dataset_generation(
                pcapng_path=input_dir,
                pcap_output_path=pcap_output_dir,
                output_split_path=output_dir,
                select_packet_len=784,
                burst_dir=burst_dir,
                start_index=28,
                is_multi=True
            )
    bigram_path = f"{output_dir}/bigram.txt"
    corpora_to_bigram(f"{output_dir}/all_biburst.txt", bigram_path)
    build_BPE(bigram_path)
    vocab_path = f"{output_dir}/vocab.txt"
    build_vocab(vocab_path)
    
    
if __name__ == "__main__":
    pretrain_data()
    