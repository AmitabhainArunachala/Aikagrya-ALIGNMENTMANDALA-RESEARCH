import json, os, platform, subprocess


def save_manifest(path: str, params: dict, results_dir: str) -> str:
    os.makedirs(results_dir, exist_ok=True)
    commit = _git_commit_sha()
    env = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    manifest = {
        "commit": commit,
        "params": params,
        "env": env,
    }
    manifest_path = os.path.join(results_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest_path


def _git_commit_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


