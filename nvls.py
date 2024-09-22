#!/bin/python

import argparse
import datetime
import logging
import re
from collections import defaultdict
from typing import List

import psutil
from packaging.version import Version

try:
    import pynvml
except ModuleNotFoundError as e:
    logging.error("pynvml needs to be installed")
    exit(-1)

# api version independent function call, see check_api_version()
COLUMN_PADDING = 4


def init():
    try:
        pynvml.nvmlInit()
    except pynvml.nvml.NVMLError_LibRmVersionMismatch as e:
        logging.fatal("Version mismatch, please reload kernel module or restart")
        exit(-1)
    except Exception as e:
        logging.fatal("Error while initializing nvml", e)
        exit(-1)
    check_api_version()


def check_api_version():
    pynvmlversion = Version(pynvml.__version__)
    if pynvmlversion <= Version("11.0.0"):
        logging.error(f"Old, unsupported version of pynvml {pynvmlversion} installed, please upgrade.")


def main(args):
    init()

    processes = get_all_processes()

    if args.usersum:
        processes = sum_processes_by_users(processes)

    # mutually exclusive with usersum, usersum has priority
    if args.devicesum and not args.usersum:
        processes = sum_processes_by_device(processes)

    if args.user:
        processes = filter_processes_by_user(processes, args.user)

    if args.device:
        processes = filter_processes_by_device(processes, args.device)

    short_numbers = args.human_numbers
    print_processes(processes, short_numbers)


def get_all_processes() -> List:
    device_count = pynvml.nvmlDeviceGetCount()
    if device_count == 0:
        logging.error("No CUDA capable GPU found")
        return []

    all_processes = []
    for deviceId in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(deviceId)
        device_processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
        processes_by_pid = sorted(device_processes, key=lambda x: x.pid)
        for process in processes_by_pid:
            process_name = "UNKNOWN"
            created_time = "UNKNOWN"
            user = "UNKNOWN"

            try:
                ps_util_process = psutil.Process(process.pid)
                process_name = ps_util_process.exe()
                created_time_ts = ps_util_process.create_time()
                created_time = datetime.datetime.fromtimestamp(created_time_ts).strftime("%Y-%m-%d %H:%M:%S")
                user = ps_util_process.username()
            except psutil.AccessDenied as e:
                # looking up info on other processes as non-root is forbidden, using the default values instead
                pass

            all_processes.append({
                "Device": deviceId,
                "PID": process.pid,
                "ProcessName": process_name,
                "User": user,
                "GPU-Mem": process.usedGpuMemory,
                "Created": created_time
            })

    return all_processes


def filter_processes_by_user(all_processes, users):
    processes = []
    for user in users:
        pattern = ".*(" + user + ").*"
        filtered_processs = [process for process in all_processes if re.match(pattern, process["User"], re.IGNORECASE)]
        processes += filtered_processs
    return processes


def filter_processes_by_device(all_processes, devices):
    return [process for process in all_processes if process["Device"] in devices]


def sum_processes_by_users(processes):
    grouped_processes = defaultdict(list)
    for process in processes:
        grouped_processes[process["User"]].append(process)

    process_summary = []
    for user, processes in grouped_processes.items():
        process_summary.append({
            "Devices": list(set(p["Device"] for p in processes)),
            "User": user,
            "NumProcesses": len(processes),
            "GPUMem": sum([p["GPU-Mem"] for p in processes if p["GPU-Mem"]]),
            "Started": min([p["Created"] for p in processes if p["Created"]])
        })

    return process_summary


def sum_processes_by_device(processes):
    grouped_processes = defaultdict(list)
    for process in processes:
        grouped_processes[process["Device"]].append(process)

    process_summary = []
    for device, processes in grouped_processes.items():
        process_summary.append({
            "Device": device,
            "Users": list(set(p["User"] for p in processes)),
            "NumProcesses": len(processes),
            "GPUMem": sum([p["GPU-Mem"] for p in processes if p["GPU-Mem"]]),
            "Started": min([p["Created"] for p in processes if p["Created"]])
        })

    return process_summary


def size_format(num):
    for unit in ("", "KB", "MB", "GB"):
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}"
        num /= 1024.0
    return num


def print_processes(all_processes, short_numbers):
    if len(all_processes) == 0:
        print("No processes found")
        return

    columns = list(all_processes[0].keys())
    column_sizes = []

    if short_numbers:
        for process in all_processes:
            for column in columns:
                if isinstance(process[column], int) and column not in ["PID", "Device", "NumProcesses"]:
                    process[column] = size_format(process[column])

    for column in columns:
        max_len = max([len(str(process[column])) for process in all_processes] + [len(column)])
        column_sizes.append(max_len + COLUMN_PADDING)

    head_line = ""
    for column_name, column_size in zip(columns, column_sizes):
        head_line += "" + column_name + (" " * (column_size - len(column_name) - 1))
    print(head_line)

    for process in all_processes:
        line = ""
        for column_name, column_size in zip(columns, column_sizes):
            line += "" + str(process[column_name]) + (" " * (column_size - len(str(process[column_name])) - 1))
        print(line)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='nvls',
                                     description='ls-like information for Nvidia GPU processes',
                                     add_help=False)
    parser.add_argument("--help",
                        action="store_true")
    parser.add_argument("-h",
                        action='store_true',
                        dest="human_numbers",
                        help="Human-readable numbers (e.g. 9k instead of 9001)")
    parser.add_argument("-d", "--device",
                        type=int,
                        action="append",
                        help="Filter processes by device ID")
    parser.add_argument("-u", "--user",
                        type=str,
                        action="append",
                        help="Filter processes by username or user id")
    parser.add_argument("--usersum",
                        action='store_true',
                        help="Sum up resources for each user")
    parser.add_argument("--devicesum",
                        action='store_true',
                        help="Sum up resources for each device")
    args = parser.parse_args()

    if args.help:
        parser.print_help()
        exit(0)

    main(args)
