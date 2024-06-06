import sys, pickle
from dawg import Dawg, DawgNode

QUERY = sys.argv[1]
with open("csw.pickle", "rb") as input_file:
    # Load the object from the file using pickle.load
    dawg = pickle.load(input_file)

if dawg.lookup(QUERY):
    print(f"{QUERY} exists in dictionary")
else:
    print(f"{QUERY} does not exist in dictionary")