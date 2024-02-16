import numpy as np


class kmer_featurization:

    def __init__(self, k):
        """
        seqs: a list of DNA sequences
        k: the "k" in k-mer
        """
        self.k = k
        self.letters = ['A', 'T', 'C', 'G']
        self.multiplyBy = 4 ** np.arange(k - 1, -1,
                                         -1)  # the multiplying number for each digit position in the k-number system
        self.n = 4 ** k  # number of possible k-mers

    def obtain_kmer_feature_for_a_list_of_sequences(self, seqs, write_number_of_occurrences=False):
        """
        Given a list of m DNA sequences, return a 2-d array with shape (m, 4**k) for the 1-hot representation of the kmer features.

        Args:
          write_number_of_occurrences:
            a boolean. If False, then in the 1-hot representation, the percentage of the occurrence of a kmer will be recorded; otherwise the number of occurrences will be recorded. Default False.
        """
        kmer_features = []
        for seq in seqs:
            this_kmer_feature = self.obtain_kmer_feature_for_one_sequence(seq,
                                                                          write_number_of_occurrences=write_number_of_occurrences)
            kmer_features.extend(this_kmer_feature)

        kmer_string = ' '.join(kmer_features)

        return kmer_string

    def obtain_kmer_feature_for_one_sequence(self, seq, write_number_of_occurrences=False):
        """
        Given a DNA sequence, return the 1-hot representation of its kmer feature.

        Args:
          seq:
            a string, a DNA sequence
          write_number_of_occurrences:
            a boolean. If False, then in the 1-hot representation, the percentage of the occurrence of a kmer will be recorded; otherwise the number of occurrences will be recorded. Default False.
        """
        number_of_kmers = len(seq) - self.k + 1

        kmers = []

        for i in range(number_of_kmers):
            this_kmer = seq[i:(i + self.k)]

            kmers.append(this_kmer)

        return kmers

# if __name__ == '__main__':
#     SEQUENCE = """ccgcgctcc tccgggcaga gcgcgtgtgg cggccgagca catgggcccg cgggccgggc gggctcgggg cggccgggac gaggaggggc gacgacgagc tgcgagcaaa gatgtgcccc gggacccccg gcaccttcca gtggatttcc ttgcggaaag gatgttggcg gtccctgtga cctgtggaga cacggccaga"""
#     featurizer = kmer_featurization(5)
#
#     # for seq in SEQUENCE.split(" "):
#     #     print(seq)
#
#     print(featurizer.obtain_kmer_feature_for_a_list_of_sequences(SEQUENCE.split(" ")))
#
