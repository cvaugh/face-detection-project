import argparse
from base64 import b64encode

import requests

import fd_wrapper as wrapper
from fd_wrapper import transform
#from fd_wrapper.detectors import *

parser = argparse.ArgumentParser(description="Automatically classify images based on  from a file.")
parser.add_argument("instructions", type=str, help="file containing instructions (one per line)")
parser.add_argument("labels", type=str, help="TSV file containing image labels")
parser.add_argument("--send", metavar="<url>", dest="send_to", const=None, default=None, action="store", nargs=1, type=str,
                    help="send the results of each operation to this URL via HTTP POST, as base64-encoded CSV")
parser.add_argument("--ident", metavar="<string>", dest="ident", const=None, default=None, action="store", nargs=1, type=str,
                    help="identification string for POST request")

args = parser.parse_args()
print(args.instructions)
print(args.labels)
send_to = None if args.send_to is None else args.send_to[0]
identity = "" if args.ident is None else args.ident[0]

def send_results(data: str):
    global send_to, identity
    if send_to is None:
        return
    r = requests.post(send_to,
                      json={"identity": identity, "data": b64encode(bytes(data, "utf-8")).decode("utf-8")},
                      headers={"Content-Type": "application/json"})
    if r.status_code != 200:
        print(f"POST failed ({r.status_code}): {r.reason}")

if __name__ != "__main__":
    exit()

paths = wrapper.get_ground_truth(args.labels, relative_to=wrapper.relative_path("./images", root=__file__))[0]

print(paths)
# to do
