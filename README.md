# Network Tomography

Scripts used for my thesis "Network Tomography & Neutrality" at School of ECE NTUA.

I created two Python scripts that implement the Rooted-Neighbor Joining (RNJ) algorithm, created by Jian Ni and Sekhar Tatikonda, for Binary and for General trees.

 The algorithm utilizes the theory of Network Tomography and Markov Random Fields to estimate the logical topology and the link parameters of a network, based solely on end-to-end, path-level traffic measurements.

 Experiments were run on the *Common Open Research Emulator (CORE)*. However, scripts can be used on any kind of emulator or real network, as long as there is a *.txt* file with the IDs of each end node and a folder with the *tcpdumps* of each one of those nodes. The dumps should be named accordingly to their corresponding node's ID.

 ## References

 * A. Coates et al. “Internet tomography”. In: IEEE Signal Processing Magazine 19.3 (2002),
pp. 47–65. issn: 1053-5888. doi: 10.1109/79.998081.
 * J. Ni and S. Tatikonda. “A Markov Random Field Approach to Multicast-Based Network
Inference Problems”. In: 2006 IEEE International Symposium on Information Theory.
2006, pp. 2769–2773. doi: 10.1109/ISIT.2006.261566.
 * Common Open Research Emulator. url: https : / / www . nrl . navy . mil / itd / ncs /
products/core.
