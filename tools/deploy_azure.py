"""Deploy the F1 Driving Lab to Azure: create the infrastructure from infra/main.bicep, then upload the website.

    az login                                            # once
    python tools/deploy_azure.py                        # infrastructure and site: Standard plan, about USD 9 a month
    python tools/deploy_azure.py --sku Free --labs v2   # the student lab only; the Free plan costs nothing
    python tools/deploy_azure.py --what-if              # show what the infrastructure step would change, then stop
    python tools/deploy_azure.py --skip-infra           # upload a new build to the site that is already there
    python tools/deploy_azure.py --verify-only          # compare the live site with the last staged build, changing nothing

In order, it
  1. checks that the Azure CLI is signed in and that Node.js is available;
  2. stages the site with tools/build_site.py, so content that is too big or unsafe is rejected before anything is created;
  3. deploys infra/main.bicep (a resource group and a Static Web App) at subscription scope;
  4. uploads the staged site with the Static Web Apps CLI. The deployment token is read from Azure and handed to the CLI in
     its environment: it is never printed, put on a command line or written to disk;
  5. downloads every file from the live site and compares it with site-manifest.json.

Remove everything again with:  az group delete --name rg-<environment> --yes
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Sequence
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "infra" / "main.bicep"
MANIFEST = ROOT / "site-manifest.json"
SITE = ROOT / "site"
SWA_CLI = "@azure/static-web-apps-cli@2.1.0"  # pinned: the upload client it downloads on first use is part of this version
# Keep in step with infra/main.bicepparam (tests/test_deploy_azure.py checks that they agree).
DEFAULTS = {"environmentName": "f1-driving-lab", "location": "westeurope", "skuName": "Standard"}
NOT_SERVED = {"staticwebapp.config.json"}  # Azure reads this as configuration and answers 404 if asked for it


class DeployError(Exception):
    """A step failed; the message says what to do about it."""


def executable(name: str, hint: str) -> str:
    path = shutil.which(name)
    if not path:
        raise DeployError(f"'{name}' was not found on PATH. {hint}")
    return path


def redact(text: str, secrets: Sequence[str]) -> str:
    for secret in secrets:
        if secret:
            text = text.replace(secret, "***")
    return text


def run(command: Sequence[str], secrets: Sequence[str] = ()) -> str:
    """Run a command and return its standard output; on failure raise with its error output (never its standard output)."""
    result = subprocess.run(list(command), capture_output=True, text=True)
    if result.returncode != 0:
        detail = redact(result.stderr.strip() or "(no error output)", secrets)
        raise DeployError(f"{Path(command[0]).stem} {command[1]} failed ({result.returncode}):\n{detail}")
    return result.stdout


def stream(command: Sequence[str], env: dict[str, str], secrets: Sequence[str]) -> None:
    """Run a long command, echoing its output with the secrets masked."""
    with subprocess.Popen(list(command), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as process:
        assert process.stdout is not None
        for line in process.stdout:
            print("  " + redact(line.rstrip(), secrets), flush=True)
    if process.returncode != 0:
        raise DeployError(f"{Path(command[0]).stem} {command[1]} failed ({process.returncode}); see the output above")


def az(*args: str, subscription: str | None = None) -> str:
    command = [executable("az", "Install the Azure CLI: https://aka.ms/installazurecli"), *args, "--only-show-errors"]
    if subscription:
        command += ["--subscription", subscription]
    return run(command)


def signed_in(subscription: str | None) -> dict:
    try:
        return json.loads(az("account", "show", "-o", "json", subscription=subscription))
    except DeployError as error:
        raise DeployError(f"Not signed in to Azure (or that subscription is not available). Run 'az login'.\n{error}") from None


def ensure_provider(namespace: str, subscription: str | None) -> None:
    state = az("provider", "show", "--namespace", namespace, "--query", "registrationState", "-o", "tsv", subscription=subscription).strip()
    if state != "Registered":
        print(f"Registering the {namespace} resource provider (once per subscription)...")
        az("provider", "register", "--namespace", namespace, "--wait", subscription=subscription)


def stage(labs: str, sku: str) -> None:
    command = [sys.executable, str(ROOT / "tools" / "build_site.py"), "--labs", labs, "--sku", sku, "--out", str(SITE), "--manifest", str(MANIFEST)]
    result = subprocess.run(command, text=True)
    if result.returncode != 0:
        raise DeployError("The site could not be staged (see above).")


def parameters(params: dict[str, str]) -> list[str]:
    return ["--parameters", *(f"{name}={value}" for name, value in params.items())]


def what_if(params: dict[str, str], subscription: str | None) -> None:
    command = [
        executable("az", "Install the Azure CLI: https://aka.ms/installazurecli"), "deployment", "sub", "what-if",
        "--name", params["environmentName"], "--location", params["location"], "--template-file", str(TEMPLATE),
        *parameters(params), "--no-prompt", "--only-show-errors",
    ]
    if subscription:
        command += ["--subscription", subscription]
    if subprocess.run(command).returncode != 0:
        raise DeployError("The what-if preview failed (see above).")


def outputs_of(text: str) -> dict[str, str]:
    outputs = {name: item["value"] for name, item in json.loads(text).items()}
    missing = {"resourceGroupName", "staticWebAppName", "staticWebAppUrl"} - set(outputs)
    if missing:
        raise DeployError(f"The deployment did not report {', '.join(sorted(missing))}.")
    return outputs


def deploy_infrastructure(params: dict[str, str], subscription: str | None) -> dict[str, str]:
    print("Deploying the infrastructure (a resource group and a Static Web App; about a minute)...")
    text = az(
        "deployment", "sub", "create", "--name", params["environmentName"], "--location", params["location"],
        "--template-file", str(TEMPLATE), *parameters(params), "--query", "properties.outputs", "-o", "json",
        subscription=subscription,
    )
    return outputs_of(text)


def existing_infrastructure(environment: str, subscription: str | None) -> dict[str, str]:
    try:
        text = az("deployment", "sub", "show", "--name", environment, "--query", "properties.outputs", "-o", "json", subscription=subscription)
    except DeployError as error:
        raise DeployError(f"No deployment named '{environment}' found; run without --skip-infra first.\n{error}") from None
    return outputs_of(text)


def deployment_token(name: str, group: str, subscription: str | None) -> str:
    for attempt in range(6):  # the key can take a few seconds to appear on a new site
        try:
            token = az("staticwebapp", "secrets", "list", "--name", name, "--resource-group", group, "--query", "properties.apiKey", "-o", "tsv", subscription=subscription).strip()
            if token:
                return token
        except DeployError:
            if attempt == 5:
                raise
        time.sleep(5)
    raise DeployError("Azure returned no deployment token for the Static Web App.")


def upload(site: Path, token: str) -> None:
    npx = executable("npx", "Install Node.js 18 or newer: https://nodejs.org")
    command = [npx, "--yes", SWA_CLI, "deploy", str(site), "--env", "production"]
    print(f"Uploading {site} with the Static Web Apps CLI (a few minutes for the full site)...")
    stream(command, {**os.environ, "SWA_CLI_DEPLOYMENT_TOKEN": token}, [token])


def fetch(url: str) -> tuple[int, bytes]:
    request = urllib.request.Request(url, headers={"Accept-Encoding": "identity", "User-Agent": "f1-driving-lab-deploy-check"})
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, b""
    except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
        return 0, str(error).encode()


def matches(data: bytes, entry: dict) -> bool:
    return len(data) == entry["bytes"] and hashlib.sha256(data).hexdigest() == entry["sha256"]


def wait_until_live(url: str, manifest: dict, timeout: float = 600.0, pause: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while True:
        status, data = fetch(url.rstrip("/") + "/")
        if status == 200 and matches(data, manifest["files"]["index.html"]):
            return
        if time.monotonic() > deadline:
            raise DeployError(f"{url} did not serve the new start page within {timeout:.0f} s (last HTTP status {status}).")
        time.sleep(pause)


def verify_site(url: str, manifest: dict, workers: int = 8, attempts: int = 4, pause: float = 5.0) -> list[str]:
    """Download every staged file from the live site; return a description of each one that is missing or differs."""
    base = url.rstrip("/")
    expected = {"/" + quote(name): entry for name, entry in manifest["files"].items() if name not in NOT_SERVED}
    expected["/"] = manifest["files"]["index.html"]

    def check(item: tuple[str, dict]) -> str | None:
        path, entry = item
        problem = ""
        for attempt in range(attempts):
            status, data = fetch(base + path)
            if status == 200 and matches(data, entry):
                return None
            problem = f"{path}: HTTP {status}" if status != 200 else f"{path}: {len(data)} bytes served, {entry['bytes']} expected (content differs)"
            if attempt + 1 < attempts:
                time.sleep(pause)
        return problem

    with ThreadPoolExecutor(workers) as pool:
        return [problem for problem in pool.map(check, expected.items()) if problem]


def verify_live(url: str) -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    print(f"Waiting for {url} to serve the new site, then comparing every file with the manifest...")
    wait_until_live(url, manifest)
    problems = verify_site(url, manifest)
    if problems:
        raise DeployError("The live site differs from the staged site:\n  " + "\n  ".join(problems))
    served = len(manifest["files"]) - len(NOT_SERVED & set(manifest["files"]))
    print(f"Verified: all {served} served files on the live site are byte for byte identical to the staged copy.")


def deploy(args: argparse.Namespace) -> None:
    executable("az", "Install the Azure CLI: https://aka.ms/installazurecli")
    if not (args.skip_site or args.verify_only):
        executable("npx", "Install Node.js 18 or newer: https://nodejs.org")
    account = signed_in(args.subscription)
    print(f"Azure subscription: {account['name']} ({account['id']}), signed in as {account['user']['name']}")
    params = {"environmentName": args.environment, "location": args.location, "skuName": args.sku}

    if args.what_if:
        what_if(params, args.subscription)
        return
    if args.verify_only:
        verify_live(existing_infrastructure(args.environment, args.subscription)["staticWebAppUrl"])
        return
    if not args.skip_site:
        stage(args.labs, args.sku)
    if args.skip_infra:
        outputs = existing_infrastructure(args.environment, args.subscription)
    else:
        ensure_provider("Microsoft.Web", args.subscription)
        outputs = deploy_infrastructure(params, args.subscription)
    url = outputs["staticWebAppUrl"]
    print(f"Static Web App {outputs['staticWebAppName']} in resource group {outputs['resourceGroupName']}: {url}")

    if not args.skip_site:
        upload(SITE, deployment_token(outputs["staticWebAppName"], outputs["resourceGroupName"], args.subscription))
        if not args.skip_verify:
            verify_live(url)

    print(f"\nInfrastructure ready at {url} (no site uploaded yet)." if args.skip_site else f"\nLive at {url}")
    print(f"Remove everything with: az group delete --name {outputs['resourceGroupName']} --yes")


def parse(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deploy the F1 Driving Lab to Azure Static Web Apps.")
    parser.add_argument("--environment", default=DEFAULTS["environmentName"], help="names the resource group (rg-NAME) and the app (swa-NAME)")
    parser.add_argument("--location", default=DEFAULTS["location"], help="region of the resource group and app (default %(default)s)")
    parser.add_argument("--sku", choices=("Free", "Standard"), default=DEFAULTS["skuName"], help="Free fits about 250 MB (--labs v2); Standard fits 500 MB, about USD 9 a month")
    parser.add_argument("--labs", default="all", help="'all' or a comma-separated list of labs, as for tools/build_site.py")
    parser.add_argument("--subscription", help="subscription name or id (default: the one 'az account show' reports)")
    parser.add_argument("--what-if", action="store_true", help="preview the infrastructure change and stop")
    parser.add_argument("--skip-infra", action="store_true", help="reuse the infrastructure from an earlier deployment")
    parser.add_argument("--skip-site", action="store_true", help="deploy the infrastructure only")
    parser.add_argument("--skip-verify", action="store_true", help="do not download the live site to compare it with the staged copy")
    parser.add_argument("--verify-only", action="store_true", help="only compare the live site with the last staged build (site-manifest.json)")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        deploy(parse(argv))
    except DeployError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
