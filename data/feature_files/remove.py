import os

features = [
    "1stIn",
    "1stWon",
    "2ndWon",
    "ace",
    "bpFaced",
    "bpSaved",
    "df",
    "h2h",
    "SvGms",
    "svpt",
]

for f in features:
    os.system("rm {0}/*".format(f))
