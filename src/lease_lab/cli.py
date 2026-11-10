import argparse
import os


def main():
    parser = argparse.ArgumentParser(
        description="Fictional lease lab; scientific work is offline and explicit."
    )
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument(
        "--through", choices=["exact", "pooled", "core"], default="exact"
    )
    sub.add_parser("serve")
    sub.add_parser("benchmark-tabpfn")
    args = parser.parse_args()
    if args.command in ("build", "benchmark-tabpfn"):
        for variable in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "BLIS_NUM_THREADS",
        ):
            os.environ[variable] = "1"
    if args.command == "benchmark-tabpfn":
        from lease_lab.evaluation.tabpfn_challenger import run_challenger

        report = run_challenger()
        print(
            f"Saved real TabPFN V2 point-only benchmark: {report.elapsed_seconds:.2f}s, {report.peak_rss_bytes / 1e6:.0f}MB peak RSS."
        )
    elif args.command == "build":
        from lease_lab.offline import build_exact, build_pooled, build_learning

        result = (
            build_exact()
            if args.through == "exact"
            else build_pooled()
            if args.through == "pooled"
            else build_learning()
        )
        print(
            f"Saved {result.provenance.run_id}: {len(result.beta.cells)} exact Beta cells. Simulation only."
        )
    elif args.command == "serve":
        import uvicorn

        uvicorn.run("lease_lab.api.app:app", host="127.0.0.1", port=43187)


if __name__ == "__main__":
    main()
