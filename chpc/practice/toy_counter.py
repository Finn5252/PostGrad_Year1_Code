"""
toy_counter.py: a practice program for learning the CHPC job cycle.

It does no real work. It "trains" by counting from 1 to --steps, sleeping a little
at each step. It also shows, in miniature, the habits your real training script needs
on a cluster:

  1. All settings come from command-line arguments (no hard-coded paths).
  2. Progress is printed with flush=True, so the log file updates while the job runs.
  3. State is checkpointed to disk regularly. Writing to a temporary file and then
     renaming it means a job killed mid-write never leaves a corrupt checkpoint.
  4. On start-up it resumes from an existing checkpoint instead of starting over.
  5. It watches its own time budget and stops cleanly before the scheduler's
     walltime limit kills it.
  6. It also catches SIGTERM, the signal PBS sends when you qdel a job or the
     walltime runs out, and saves before exiting.
  7. It writes a DONE marker when the work is finished, so a job script can tell
     "finished" from "needs another job".

Only the Python standard library is used, so it runs on any Python 3.6+ without
setting up an environment.
"""
import argparse
import json
import os
import signal
import socket
import sys
import time


def parse_args():
    p = argparse.ArgumentParser(description="Toy resumable job for CHPC practice")
    p.add_argument("--steps", type=int, default=120, help="total steps to count")
    p.add_argument("--sleep", type=float, default=5.0, help="seconds per step")
    p.add_argument("--out-dir", required=True, help="where checkpoint/log/results go")
    p.add_argument("--checkpoint-every", type=int, default=5, help="steps between checkpoints")
    p.add_argument(
        "--time-budget-min", type=float, default=None,
        help="stop cleanly after this many minutes (set a little below the job walltime)",
    )
    return p.parse_args()


def save_checkpoint(path, state):
    """Atomic write: write a temp file, then rename it over the old one."""
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(state, fh)
        fh.flush()
        os.fsync(fh.fileno())  # make sure the bytes really reach the disk
    os.replace(tmp, path)      # rename is atomic: the old or the new file, never half of one


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_path = os.path.join(args.out_dir, "checkpoint.json")
    done_path = os.path.join(args.out_dir, "DONE")

    if os.path.exists(done_path):
        print("[toy] DONE marker exists, nothing to do.", flush=True)
        return 0

    # Resume or start fresh
    if os.path.exists(ckpt_path):
        with open(ckpt_path) as fh:
            state = json.load(fh)
        print("[toy] resuming from step {}".format(state["step"]), flush=True)
    else:
        state = {"step": 0, "total": 0, "n_jobs": 0}
        print("[toy] starting fresh", flush=True)
    state["n_jobs"] += 1

    print("[toy] host={} pid={} PBS_JOBID={}".format(
        socket.gethostname(), os.getpid(), os.environ.get("PBS_JOBID", "<not in a PBS job>")),
        flush=True)

    # SIGTERM handler: PBS sends SIGTERM on qdel or walltime, then SIGKILL a few seconds
    # later. We only set a flag here; the loop checks it and saves cleanly.
    stop_requested = {"flag": False}

    def on_sigterm(signum, frame):
        print("[toy] received SIGTERM, will checkpoint and exit", flush=True)
        stop_requested["flag"] = True

    signal.signal(signal.SIGTERM, on_sigterm)

    t_start = time.time()
    budget_s = args.time_budget_min * 60 if args.time_budget_min else None

    while state["step"] < args.steps:
        if budget_s is not None and time.time() - t_start > budget_s:
            print("[toy] time budget used up at step {}, saving and exiting".format(state["step"]),
                  flush=True)
            break
        if stop_requested["flag"]:
            break

        time.sleep(args.sleep)          # the "work"
        state["step"] += 1
        state["total"] += state["step"]
        print("[toy] step {}/{} total={}".format(state["step"], args.steps, state["total"]),
              flush=True)

        if state["step"] % args.checkpoint_every == 0:
            save_checkpoint(ckpt_path, state)

    save_checkpoint(ckpt_path, state)

    if state["step"] >= args.steps:
        with open(os.path.join(args.out_dir, "result.json"), "w") as fh:
            json.dump({"final_total": state["total"], "jobs_used": state["n_jobs"]}, fh, indent=2)
        open(done_path, "w").close()
        print("[toy] finished: total={} using {} job(s)".format(state["total"], state["n_jobs"]),
              flush=True)
    else:
        print("[toy] not finished yet ({}/{}); submit again to resume".format(
            state["step"], args.steps), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
