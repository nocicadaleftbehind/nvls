#!/bin/python

import argparse
import datetime
import re
from collections import defaultdict

import psutil
from packaging.version import Version

try:
    import pynvml
except ModuleNotFoundError as e:
    print("pynvml needs to be installed")
    exit(0)

# api version independent function call, see check_api_version()
getProcessesFunction = lambda x: x

COLUMN_PADDING = 4


def init():
    pynvml.nvmlInit()
    check_api_version()


def check_api_version():
    # there are multiple pynvml version, we need to call the correct version based on our installed pynvml version
    global getProcessesFunction

    pynvmlversion = Version(pynvml.__version__)
    if pynvmlversion >= Version("11.5.0"):
        api_version = "v3"
    else:
        api_version = "v2"

    if api_version == "v2":
        getProcessesFunction = pynvml.nvmlDeviceGetComputeRunningProcesses_v2
    if api_version == "v3":
        getProcessesFunction = pynvml.nvmlDeviceGetComputeRunningProcesses_v3


def main(args):
    init()

    deviceCount = pynvml.nvmlDeviceGetCount()

    print_system_info(deviceCount)

    processes = get_all_processes(deviceCount)

    if args.usersum:
        processes = sum_processes_by_users(processes)

    if args.devicesum:
        processes = sum_processes_by_device(processes)

    if args.user:
        processes = filter_processes_by_user(processes, args.user)

    short_numbers = args.human_numbers
    print_processes(processes, short_numbers)


def get_all_processes(device_count):
    all_processes = []
    for deviceId in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(deviceId)
        device_processes = getProcessesFunction(handle)
        processes_by_pid = sorted(device_processes, key=lambda x: x.pid)
        for process in processes_by_pid:
            user = "UNKNOWN"
            process_name = "UNKNOWN"
            created_time = "UNKNOWN"

            try:
                ps_util_process = psutil.Process(process.pid)
                user = ps_util_process.username()
                process_name = ps_util_process.exe()
                created_time_ts = ps_util_process.create_time()
                created_time = datetime.datetime.fromtimestamp(created_time_ts).strftime("%Y-%m-%d %H:%M:%S")
            except psutil.AccessDenied as e:
                # looking up info on other processes as non-root is forbidden
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
    pattern = ".*(" + "|".join(users) + ").*"
    return [process for process in all_processes if re.match(pattern, process["Username"], re.IGNORECASE)]


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


def print_system_info(deviceCount):
    print(f"Cuda version: {pynvml.nvmlSystemGetCudaDriverVersion_v2()}")
    print(f"Driver version: {pynvml.nvmlSystemGetDriverVersion()}")

    for deviceId in range(deviceCount):
        handle = pynvml.nvmlDeviceGetHandleByIndex(deviceId)
        print(f"Device {deviceId}: {pynvml.nvmlDeviceGetName(handle)}")

    print()


def size_format(num):
    for unit in ("", "KB", "MB", "GB"):
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}"
        num /= 1024.0
    return num


def print_processes(all_processes, short_numbers):
    columns = list(all_processes[0].keys())
    column_sizes = []

    #if short_numbers:
    if True:
        for process in all_processes:
            for column in columns:
                if isinstance(process[column], int) and column not in ["PID", "Device"]:
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
                                     description='ls-like information for Nvidia GPU processes')
    parser.add_argument("-u", "--user",
                        type=str,
                        nargs="*")
    parser.add_argument("-s",
                        dest="human_numbers",
                        action='store_true',
                        help="Human-readable numbers (9k instead of 9001)")
    parser.add_argument("--usersum",
                        action='store_true')
    parser.add_argument("--devicesum",
                        action='store_true')
    args = parser.parse_args()
    main(args)
