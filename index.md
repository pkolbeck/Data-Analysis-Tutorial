# Investigating the Properties of Central Proteins in an Interaction Network
By: Paul Kolbeck

Proteins are large molecules inside of organisms that fulfill a huge variety of functions within organisms. All of biological life hinges on the functions of proteins within organisms. From viruses to whales and everything in between, living beings are basically a collection of proteins that, through complex interactions and mechanisms, ensure the continuation of their own existence. As proteins are an underlying part of all biological processes, understanding proteins and their interactions is an important portion research in biology. 

Proteins are created when a ribosome (a type of protein complex) takes in a gene sequence in the from of a strand of messenger RNA (mRNA), which was transcripted from the DNA in the nucleus of the cell, and starts an amino acid chain. Amino acids are the building blocks of proteins, and are small organic compounds that have binding sites on them that allow them to be bound together in a chain. As the ribosome travels along the mRNA strand, it encounters a sequence of triplets of bases, known as codons. Each codon corresponds to an amino acid, which the ribosome then takes from the surrounding supply and attaches to the budding amino acid chain in the order in which the codons appear. Thus, the mRNA strand is used to build a new sequence of amino acids. This sequence of amino acids then folds into a 3d shape, which then has complex properties based on the sequence of amino acids and how it folded. 

Not only are the proteins themselves complex, but they exist within a sea of other proteins with which they may interact, giving rise to a network of interactions which further infuence the functionality of proteins. As the environments change, so does the function of a protein, so a protein that is a part of one set of interactions in a liver that allow the liver to grow back, may be part of a different set of interactions in skeletal muscle that allow the muscle to contract. Mapping these interactions is thus another integral part in understanding the biological processes in an organism.

In this tutorial, our goal is to investigate possible relationships between several protein characteristics, particularly their role in an interaction network and their structure. For example, one might predict that the number of pathways a protein is a part of may correlate to the number of active sites on a protein, which further correlates with the mass (i.e. size). TODO

## Tools
For this analysis, we use Python 3, along with external packages numpy, pandas, matplotlib, networkx, and sklearn. TODO

## Gathering and Cleaning Data
For this tutorial, we chose to use the interaction network data provided by [Innatedb](https://www.innatedb.com/). Innatedb is a database of genes, interactions, and signaling pathways related to the innate immune response of human beings. This was chosen so as to provide a sample of the protein interaction network in the human body with a fleshed out interaction network. As this is a biased sample of the population of all proteins in the human body, results derived from the analysis will only definitively apply to this subset of data, but may inform further investigation in other portions of the human protein interaction network. 

### Acquiring the Innatedb Data
The InnateDB database is only on the order of a hundred Megabytes, so downloading and importing the entire database was taken as the easiest and cleanest way to work with the database. 
