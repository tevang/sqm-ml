#!/bin/env python

from argparse import ArgumentParser, RawDescriptionHelpFormatter
from collections import defaultdict

## Parse command line arguments
def cmdlineparse():
    parser = ArgumentParser(formatter_class=RawDescriptionHelpFormatter, description="""
DESCRIPTION:
    find the common molnames with bioactivity among all provided files and writes them into file 'common_activities.txt'.
                            """,
                            epilog="""""")
    parser.add_argument("-activities", dest="ACTIVITIES_FILE_LIST", required=False, action='append', default = [],
                        help="A file with the bioactivities of the compounds. It must contain two columns, the "
                             "first must have the molname and the second 0/1 for inactive/active, respectivelly."
                             " This argument can be used multiple times.")

if __name__ == '__main__':
    args = cmdlineparse()
    molnamesSet_dict = defaultdict(set)
    all_activities_dict = {}
    for activity_file in args.ACTIVITIES_FILE_LIST:
        activities_dict = {l.split()[0].lower(): int(l.split()[1]) for l in open(activity_file, 'r')}
        molnamesSet_dict[activity_file].add(activities_dict.keys())
        for k,v in activities_dict.items():
            all_activities_dict[k] = v

    common_molnames_set = list(molnamesSet_dict.values())[1]
    for molnames_set in list(molnamesSet_dict.values())[1:]:
        common_molnames_set = common_molnames_set.intersection(molnames_set)

    with open("common_activities.txt", 'w') as f:
        for molname in common_molnames_set:
            f.write("%s %i\n" % (molname, all_activities_dict[molname]))

