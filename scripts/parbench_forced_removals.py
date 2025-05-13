#!/usr/bin/env python3
import os
import subprocess
import signal
import sys
import time
import csv
import argparse


def run_binary_with_files(
    input_dir, binary_path, output_dir, timeout, num_threads, kill_buffer, num_rules
):
    # Initialize counters and process list
    total_files = 0
    processed_files = 0
    processes = []

    # Create a stack of all files
    file_stack = []
    for filename in os.listdir(input_dir):
        file_path = os.path.join(input_dir, filename)
        if os.path.isfile(file_path):
            for i in range(num_rules):
                file_stack.append((i, file_path))
                total_files += 1
    file_stack.sort(reverse=True)
    output_stack = []

    header = ",".join(
        [
            "name",
            "runtime",
            "read_time_ms",
            "greedy_time_ms",
            "init_time_ms",
            "num_iter",
            "iter_time_ms",
            "domset_size",
        ]
    )

    # Loop until all files are processed and no active processes remain
    while file_stack or processes:
        # Check if any processes have terminated
        for i, (
            process,
            graph_file_path,
            rule_id,
            start_time,
            sigterm_flag,
        ) in enumerate(processes):
            if process.poll() is not None:
                end_time = time.time()
                runtime = end_time - start_time
                for line in process.stdout:
                    [rtms, gtms, itms, ni, iterms, ds] = (
                        line.decode("utf-8").strip().split(",")
                    )
                    output_stack.append(
                        (
                            rule_id,
                            [
                                os.path.basename(graph_file_path),
                                runtime,
                                rtms,
                                gtms,
                                itms,
                                ni,
                                iterms,
                                ds,
                            ],
                        )
                    )
                    print(
                        f"Result {graph_file_path} -- {rule_id} -- {runtime} -- {rtms}ms -- {gtms}ms -- {itms}ms -- {ni} -- {iterms}ms -- {ds}"
                    )
                    break
                processes.pop(i)
                processed_files += 1
                print(f"Process terminated with runtime: {runtime:.2f} seconds")

        # Replace terminated processes with new ones (if possible)
        while len(processes) < num_threads and file_stack:
            (rule_id, file_path) = file_stack.pop()
            cmd = [binary_path, "--path", file_path, "--forced", str(rule_id)]
            process_start_time = time.time()
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                preexec_fn=os.setsid,
            )
            processes.append(
                (process, file_path, rule_id, process_start_time, False)
            )

        # Check timeout and terminate or kill processes
        for i in range(len(processes)):
            process, graph_file_path, rule_id, start_time, sigterm_flag = processes[i]
            elapsed_time = time.time() - start_time
            if elapsed_time > timeout:
                if sigterm_flag and elapsed_time > timeout + kill_buffer:
                    print("Kill proc")
                    process.kill()
                    output_stack.append(
                        (
                            rule_id,
                            [
                                os.path.basename(graph_file_path),
                                "-",
                                "-",
                                "-",
                                "-",
                                "-",
                                "-",
                                "-",
                            ],
                        )
                    )
                elif not sigterm_flag:
                    process.terminate()
                    processes[i] = (process, graph_file_path, rule_id, start_time, True)

        # Sleep for a second before checking process status again
        time.sleep(0.3)
    output_stack.sort(key=lambda x: x[1][0])

    outfiles = []
    for i in range(num_rules):
        outfiles.append(open(f"{output_dir}/rule_{i}.csv", "w", newline=""))

    writers = []
    for i in range(num_rules):
        writer = csv.writer(outfiles[i])
        writer.writerow(
            [
                "name",
                "runtime",
                "read_time_ms",
                "greedy_time_ms",
                "init_time_ms",
                "num_iter",
                "iter_time_ms",
                "domset_size",
            ]
        )

        writers.append(writer)

    # Why do we need this?
    seen = {}
    for i, x in output_stack:
        if (i, x[0]) not in seen:
            seen[(i, x[0])] = 1
            writers[i].writerow(x)

    # Print overall process information
    print(f"All processes terminated.")


def main(args):
    input_dir = args.input
    binary_path = args.binary
    output_dir = args.output
    timeout = args.timeout
    num_threads = args.num_threads
    kill_buffer = args.kill_buffer
    num_rules = args.rules

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    run_binary_with_files(
        input_dir, binary_path, output_dir, timeout, num_threads, kill_buffer, num_rules
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, help="Input directory")
    parser.add_argument("-o", "--output", required=True, help="Output directory")
    parser.add_argument(
        "-t",
        "--timeout",
        type=int,
        default=300,
        help="Timeout in seconds (default: 300)",
    )
    parser.add_argument(
        "-n",
        "--num_threads",
        type=int,
        default=4,
        help="Number of threads (default: 4)",
    )
    parser.add_argument(
        "-k",
        "--kill_buffer",
        type=int,
        default=5,
        help="Kill buffer in seconds (default: 5)",
    )
    parser.add_argument(
        "--binary", type=str, default=False, required=True, help="Binary path of solver"
    )
    parser.add_argument(
        "-r",
        "--rules",
        type=int,
        default=8,
        help="Number of ForcedRemoval-Rules (default: 8)",
    )
    args = parser.parse_args()
    main(args)
