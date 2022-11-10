"""
Converts test results from JSON to CSV for ease of analysis.
"""

import argparse
import json


def convert_results(infile, outfile):
    statistics = {'passed with default': 0, 'passed with plan': 0, 'failed with plan': 0, 'failed without plan': 0}

    with open(infile, "r") as in_file:
        results = json.load(in_file)
        for entry in results['levels']:
            if entry['status'] == 'Pass':
                if entry.get('Default action used'):
                    statistics['passed with default'] += 1
                else:
                    statistics['passed with plan'] += 1
            else:
                if entry.get('Default action used'):
                    statistics['failed without plan'] += 1
                else:
                    statistics['failed with plan'] += 1

    print(statistics)

    with open(infile, "r") as in_file:
        results = json.load(in_file)
        if not outfile:
            outfile = infile + ".csv"
        with open(outfile, "w+") as outfile:
            outfile.write('birds remaining,birds start,level,pig remaining,pigs start,simplification level 1 time,'
                          'simplification level 2 time,repair calls,repair description, score,status\n')
            for entry in results['levels']:
                outfile.write(f'{entry["birds_remaining"]},{entry["birds_start"]},{entry["level"]},'
                              f'{entry["pigs_remaining"]},{entry["pigs_start"]},'
                              f'{entry["simplification level time 1"] if entry.get("simplification level time 2") else ""},'
                              f'{entry["simplification level time 2"] if entry.get("simplification level time 2") else ""},'
                              f'{" ".join(entry["repair description"]).replace(",", ":") if entry.get("repair description") else ""},'
                              f'{entry["repair_calls"] if entry.get("repair_calls") else ""},'
                              f'{entry["score"]},{entry["status"]}\n')


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description="Converts test results from JSON to CSV for ease of analysis.")
    # parser.add_argument("infile", help="JSON file to read")
    # parser.add_argument("-o", "--outfile", help="Output file name, defaults to <infile>.csv", default=False)
    # args = parser.parse_args()
    for from_num in range(4, 270, 25):
        infile = f'stats_helpful_actions_novelty0_type246_agentRepairingHydra_from_level{from_num}.json'
        convert_results(infile, False)
