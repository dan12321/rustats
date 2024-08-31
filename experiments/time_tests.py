#!/usr/bin/python3
import argparse
import subprocess
import sys
from io import FileIO

from dataclasses import dataclass

parser = argparse.ArgumentParser(
    prog="time_tests",
    epilog="./experiments/time_tests.py -n 50 agg measurement ../1brc/1b.csv -c "';'" -g station",
    description="Script for timing runs of the rustats cli",
)

parser.add_argument("-n", "--number",
                    help= "number of times to run the command",
                    type=int,
                    required=True)
parser.add_argument("-s", "--skip-build",
                    dest="skip_build",
                    help="Don't rebuild rustats before running",
                    action="store_true")
parser.add_argument("cli_args",
                    nargs=argparse.REMAINDER,
                    help="args to pass to rustats cli are required")

def main():
    args = parser.parse_args()
    stats = call_cli(args.number, args.cli_args)
    rows = [Stats.csv_header()] + [stat.csv() for stat in stats]
    print("\n".join(rows))

class Stats:
    def __init__(self, lines: list[str]):
        for line in lines:
            parts = line.split(":")
            if len(parts) < 2:
                continue
            match parts[0].strip():
                case "User time (seconds)":
                    self.user_time = float(parts[1].strip())
                case "System time (seconds)":
                    self.system_time = float(parts[1].strip())
                case "Percent of CPU this job got":
                    self.percent_cpu_job_got = int(parts[1].strip()[:-1])
                case "Elapsed (wall clock) time (h":
                    # Full would be Elapsed (wall clock) time (h:mm:ss or m:ss)
                    clock_time = None
                    if len(parts) == 6:
                        clock_time = 60 * float(parts[4].strip()) + \
                            float(parts[5].strip())
                    elif len(parts) == 7:
                        clock_time = 60*60 * float(parts[4].strip()) + \
                            60 * float(parts[5].strip()) + \
                            float(parts[6].strip())
                    self.real_time = clock_time
                case "Average shared text size (kbytes)":
                    self.avg_shared_text_size = int(parts[1].strip())
                case "Average unshared data size (kbytes)":
                    self.avg_unshared_data_size = int(parts[1].strip())
                case "Average stack size (kbytes)":
                    self.avg_stack_size = int(parts[1].strip())
                case "Average total size (kbytes)":
                    self.avg_total_size = int(parts[1].strip())
                case "Maximum resident set size (kbytes)":
                    self.max_rss_mem = int(parts[1].strip())
                case "Average resident set size (kbytes)":
                    self.avg_rss_mem = int(parts[1].strip())
                case "Major (requiring I/O) page faults":
                    self.major_page_faults = int(parts[1].strip())
                case "Minor (reclaiming a frame) page faults":
                    self.minor_page_faults = int(parts[1].strip())
                case "Voluntary context switches":
                    self.voluntary_context_switches = int(parts[1].strip())
                case "Involuntary context switches":
                    self.involuntary_context_switches = int(parts[1].strip())
                case "Swaps":
                    self.swaps = int(parts[1].strip())
                case "File system inputs":
                    self.file_system_inputs = int(parts[1].strip())
                case "File system outputs":
                    self.file_system_outputs = int(parts[1].strip())
                case "Socket messages sent":
                    self.socket_messages_sent = int(parts[1].strip())
                case "Socket messages received":
                    self.socket_messages_received = int(parts[1].strip())
                case "Signals delivered":
                    self.signals_delivered = int(parts[1].strip())
                case "Page size (bytes)":
                    self.page_size = int(parts[1].strip())
                case "Exit status":
                    self.exit_status = int(parts[1].strip())
    
    def csv_header() -> str:
        return "avg_rss_mem,avg_shared_text_size,avg_stack_size,avg_total_size,avg_unshared_data_size,exit_status,file_system_inputs,file_system_outputs,involuntary_context_switches,major_page_faults,max_rss_mem,minor_page_faults,page_size,percent_cpu_job_got,real_time,signals_delivered,socket_messages_received,socket_messages_sent,swaps,system_time,user_time,voluntary_context_switches"
    
    def csv(self) -> str:
        return f"{self.avg_rss_mem},{self.avg_shared_text_size},{self.avg_stack_size},{self.avg_total_size},{self.avg_unshared_data_size},{self.exit_status},{self.file_system_inputs},{self.file_system_outputs},{self.involuntary_context_switches},{self.major_page_faults},{self.max_rss_mem},{self.minor_page_faults},{self.page_size},{self.percent_cpu_job_got},{self.real_time},{self.signals_delivered},{self.socket_messages_received},{self.socket_messages_sent},{self.swaps},{self.system_time},{self.user_time},{self.voluntary_context_switches}"


def call_cli(n: int, cli_args: list[str]) -> list[Stats]:
    result = []
    cmd = ["/usr/bin/time", "-v", "./target/release/cli"] + cli_args
    # TODO: add an option for the number of processes that can be run at 1 time
    for _ in range(n):
        with subprocess.Popen(
            cmd,
            text=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE) as p:

            stats = Stats(p.stderr.readlines())
            result.append(stats)
    return result



if __name__ == "__main__":
    main()