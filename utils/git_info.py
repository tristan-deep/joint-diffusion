"""Git utilities
Author(s): Tristan Stevens, Iris Huijben
"""

import subprocess
import sys
import warnings


def get_git_commit_hash():
    """Gets git commit hash of current branch."""
    return str(subprocess.check_output(["git", "rev-parse", "HEAD"]).strip(), "utf-8")


def get_git_branch():
    """Get current branch name."""
    return str(
        subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).strip(),
        "utf-8",
    )


def get_git_summary(verbose=False):
    """Get summary of git info.
    Args:
        verbose (bool, optional): print git summary. Defaults to False.
    Returns:
        str: git summary string.
            contains branch name and commit hash.
    """
    try:
        git_summary = get_git_branch() + "=" + get_git_commit_hash()
        if verbose:
            print("Git branch and commit: ")
            print(git_summary)
        return git_summary
    except Exception:
        warnings.warn("Cannot find Git")


if __name__ == "__main__":
    get_git_summary()
    sys.stdout.flush()
