"""Combined TabPFN model downloader with fallback options."""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path
from typing import List

from tabpfn.model_loading import _user_cache_dir


def download_with_curl(base_url: str, model_files: List[str], download_dir: Path) -> tuple[int, int]:
    """Download models using curl from Google Storage (Script 2 approach).

    Returns:
        tuple: (success_count, fail_count)
    """
    logger = logging.getLogger(__name__)
    success_count = 0
    fail_count = 0

    logger.info(f"Attempting to download {len(model_files)} models using curl from Google Storage...")
    logger.info(f"Base URL: {base_url}")
    logger.info(f"Target directory: {download_dir}")
    logger.info("-" * 60)

    for file_name in model_files:
        file_url = f"{base_url}/{file_name}"
        dest_path = download_dir / file_name

        logger.info(f"Processing {file_name}...")

        # Check if file already exists
        if dest_path.exists():
            logger.info(f"✓ File already exists: {dest_path}")
            success_count += 1
            continue

        # Check if URL is accessible using curl HEAD request
        try:
            result = subprocess.run(
                ["curl", "-LsfI", "-o", "/dev/null", file_url], check=False, capture_output=True, timeout=30
            )

            if result.returncode == 0:
                logger.info(f"URL accessible. Downloading {file_name}...")

                # Download the file
                result = subprocess.run(
                    ["curl", "-L", "-o", str(dest_path), file_url],
                    check=False,
                    capture_output=True,
                    timeout=300,  # 5 minutes timeout for download
                )

                if result.returncode == 0:
                    logger.info(f"✓ Successfully downloaded {file_name}")
                    success_count += 1
                else:
                    logger.error(f"✗ Failed to download {file_name} (curl exit code: {result.returncode})")
                    fail_count += 1
            else:
                logger.error(f"✗ File not accessible at {file_url}")
                fail_count += 1

        except subprocess.TimeoutExpired:
            logger.error(f"✗ Timeout while processing {file_name}")
            fail_count += 1
        except Exception as e:
            logger.error(f"✗ Error processing {file_name}: {e}")
            fail_count += 1

        logger.info("-" * 40)

    return success_count, fail_count


def download_with_tabpfn_builtin(cache_dir: Path, timeout: int = 600) -> bool:
    """Download models using TabPFN's built-in function (Script 1 approach).

    Args:
        cache_dir: Directory to save models
        timeout: Timeout in seconds for the download operation (default: 10 minutes)

    Returns:
        bool: True if successful, False otherwise
    """
    logger = logging.getLogger(__name__)

    try:
        logger.info(f"Attempting to download models using TabPFN's built-in function (timeout: {timeout}s)...")

        # Use subprocess with timeout to handle the download
        import subprocess
        import sys

        # Create a temporary script to run the download with timeout
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                f"from tabpfn.model_loading import download_all_models; "
                f"from pathlib import Path; "
                f"download_all_models(Path('{cache_dir}'))",
            ],
            check=False,
            timeout=timeout,
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            logger.info("✓ TabPFN built-in download completed successfully")
            return True
        else:
            logger.error(f"✗ TabPFN built-in download failed with return code {result.returncode}")
            if result.stderr:
                logger.error(f"Error output: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        logger.error(f"✗ TabPFN built-in download timed out after {timeout} seconds")
        return False
    except Exception as e:
        logger.error(f"✗ TabPFN built-in download failed: {e}")
        return False


def main() -> None:
    """Combined TabPFN model downloader with fallback options."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Download all TabPFN models with fallback options.")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Optional path to override the default cache directory.",
    )
    parser.add_argument(
        "--method",
        choices=["curl", "builtin", "auto"],
        default="auto",
        help="Download method: 'builtin' for TabPFN function, 'curl' for Google Storage, 'auto' tries builtin first then curl fallback (default: auto)",
    )
    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger(__name__)

    # Determine cache directory
    cache_dir = args.cache_dir or _user_cache_dir(platform=sys.platform, appname="tabpfn")
    cache_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Combined TabPFN Model Downloader")
    logger.info("=" * 60)
    logger.info(f"Cache directory: {cache_dir}")
    logger.info(f"Download method: {args.method}")
    logger.info("=" * 60)

    # Configuration for Google Storage download (Script 2)
    base_url = "https://storage.googleapis.com/tabpfn-v2-model-files/05152025"
    model_files = [
        "tabpfn-v2-classification-gn2p4bpt.ckpt",
        "tabpfn-v2-classification-vutqq28w.ckpt",
        "tabpfn-v2-classification-znskzxi4.ckpt",
        "tabpfn-v2-classifier-gn2p4bpt.ckpt",
        "tabpfn-v2-classifier-llderlii.ckpt",
        "tabpfn-v2-classifier-od3j1g5m.ckpt",
        "tabpfn-v2-classifier-vutqq28w.ckpt",
        "tabpfn-v2-classifier-znskzxi4.ckpt",
        "tabpfn-v2-classifier.ckpt",
        "tabpfn-v2-regressor-09gpqh39.ckpt",
        "tabpfn-v2-regressor-2noar4o2.ckpt",
        "tabpfn-v2-regressor-5wof9ojf.ckpt",
        "tabpfn-v2-regressor-wyl4o83o.ckpt",
        "tabpfn-v2-regressor.ckpt",
    ]

    success = False

    if args.method == "curl":
        # Use only curl method
        success_count, fail_count = download_with_curl(base_url, model_files, cache_dir)
        success = fail_count == 0

    elif args.method == "builtin":
        # Use only TabPFN built-in method
        success = download_with_tabpfn_builtin(cache_dir)

    else:  # args.method == "auto"
        # Try TabPFN built-in first (Method 1), then fallback to Google Storage curl (Method 2)
        logger.info("Using auto mode: trying TabPFN built-in first, then Google Storage as fallback")

        builtin_success = download_with_tabpfn_builtin(cache_dir, timeout=600)  # 10 minute timeout

        if not builtin_success:
            logger.info("\n⚠️  TabPFN built-in download failed")
            logger.info("Falling back to Google Storage download method...")
            logger.info("=" * 60)

            success_count, fail_count = download_with_curl(base_url, model_files, cache_dir)
            success = fail_count == 0
        else:
            success = True

    # Final summary
    logger.info("=" * 60)
    logger.info("Download Summary")
    logger.info("=" * 60)

    if success:
        logger.info("✅ All TabPFN models downloaded successfully!")

        # List downloaded files
        model_files_in_cache = list(cache_dir.glob("*.ckpt"))
        if model_files_in_cache:
            logger.info(f"📁 Found {len(model_files_in_cache)} model files in cache:")
            for model_file in sorted(model_files_in_cache):
                size_mb = model_file.stat().st_size / (1024 * 1024)
                logger.info(f"   - {model_file.name} ({size_mb:.1f} MB)")
    else:
        logger.error("❌ Download process completed with errors")
        logger.error("💡 Try running with --method builtin or check your internet connection")

    logger.info(f"📂 Models saved to: {cache_dir}")


if __name__ == "__main__":
    main()
