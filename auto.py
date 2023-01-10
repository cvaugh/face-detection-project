import argparse
import csv
import os
import time
from base64 import b64encode
from sys import stderr

import requests

import fd_wrapper as wrapper
from fd_wrapper import transform
from fd_wrapper.detectors import *

parser = argparse.ArgumentParser(description="Automatically classify images based on  from a file.")
parser.add_argument("instructions", type=str, help="file containing instructions (one per line)")
parser.add_argument("labels", type=str, help="TSV file containing image labels")
parser.add_argument("--send", metavar="<url>", dest="send_to", const=None, default=None, action="store", nargs=1, type=str,
                    help="send the results of each operation to this URL via HTTP POST, as base64-encoded CSV")
parser.add_argument("--ident", metavar="<string>", dest="ident", const=None, default=None, action="store", nargs=1, type=str,
                    help="identification string for POST request")
parser.add_argument("--delay", metavar="<seconds>", dest="delay", const=None, default=None, action="store", nargs=1, type=int,
                    help="wait this many seconds between instructions")
parser.add_argument("--debug", dest="debug", action="store_true",
                    help="print results instead of sending")

args = parser.parse_args()
send_to = None if args.send_to is None else args.send_to[0]
identity = "" if args.ident is None else args.ident[0]
delay = 0 if args.delay is None else args.delay[0]

def human_size(bytes, units=['B','KB','MB','GB','TB', 'PB', 'EB']):
    """
    Returns a human readable string representation of bytes.
    
    Source: https://stackoverflow.com/a/43750422
    """
    return str(bytes) + " " + units[0] if bytes < 1024 else human_size(bytes >> 10, units[1:])

def send_results(data: str):
    global send_to, identity, args
    if send_to is None:
        return
    if not isinstance(data, str):
        data = str(data)
    if args.debug:
        print(data)
        print(f"Sent {human_size(len(data))} to '{send_to}' [DEBUG]")
    else:
        r = requests.post(send_to,
                          json={"identity": identity, "data": b64encode(bytes(data, "utf-8")).decode("utf-8")},
                          headers={"Content-Type": "application/json"})
        if r.status_code == 200:
            print(f"Sent {human_size(len(data))} to '{send_to}'")
        else:
            print(f"POST failed ({r.status_code}): {r.reason}")

if __name__ != "__main__":
    exit()

paths = ["./images/" + path for path in wrapper.get_ground_truth(args.labels, relative_to=wrapper.relative_path("./images", root=__file__))[0]]

instructions = []

with open(args.instructions, "r") as file:
    next(file)
    for row in csv.reader(file):
        instructions.append({
            "transform": getattr(transform, row[0]),
            "start": int(row[1]),
            "end": int(row[2])
        })

if(len(instructions) == 0):
    print("Error: No instructions provided", file=stderr)
    exit()

for i in range(len(instructions)):
    inst = instructions[i]
    print(f"Instruction {i + 1} of {len(instructions)}: {inst['transform'].__name__} ({inst['start']} to {inst['end']})")
    results_dir = f"results/final/{inst['transform'].__name__}"
    wrapper.classify_sets(
        paths,
        batch_size=256,
        detectors=[
            blazeface,
            insightface,
            mtcnn,
            ssd
        ],
        transform=inst["transform"],
        start_index=inst["start"],
        set_count=inst["end"],
        truth_override=True,
        results_dir=results_dir
    )
    out = ""
    results = sorted(os.listdir(results_dir), key=lambda path: int(os.path.splitext(path)[0]))
    for r in results:
        path = results_dir + "/" + r
        with open(path, "r", encoding="utf-8") as file:
            out += path + "\n"
            out += "".join(file.readlines()) + "\n"
    send_results(out.strip())
    if delay > 0 and i != len(instructions) - 1:
        print("Waiting", delay, "seconds")
        time.sleep(delay)
