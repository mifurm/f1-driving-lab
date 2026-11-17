"""The Azure deployment script: its defaults match the Bicep, the token never reaches a command line, and the live site is checked byte for byte.

No Azure access is needed: the Azure CLI and the upload are replaced, and the site check runs against a local web server.
"""

import contextlib
import functools
import hashlib
import http.server
import io
import json
import os
import re
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from tools import deploy_azure
from tools.deploy_azure import DeployError

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = json.dumps({
    "resourceGroupName": {"type": "String", "value": "rg-demo"},
    "staticWebAppName": {"type": "String", "value": "swa-demo"},
    "staticWebAppUrl": {"type": "String", "value": "https://demo.example"},
})


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


class InfrastructureTests(unittest.TestCase):
    def test_script_defaults_match_the_bicep_parameter_file_and_the_template_declares_every_parameter(self):
        values = dict(re.findall(r"^param (\w+) = '([^']*)'", (ROOT / "infra" / "main.bicepparam").read_text(), re.M))
        self.assertEqual(values, deploy_azure.DEFAULTS)
        declared = set(re.findall(r"^param (\w+) ", (ROOT / "infra" / "main.bicep").read_text(), re.M))
        self.assertTrue(set(deploy_azure.DEFAULTS) <= declared)

    def test_the_template_deploys_one_resource_group_and_one_static_web_app_at_subscription_scope(self):
        main = (ROOT / "infra" / "main.bicep").read_text()
        module = (ROOT / "infra" / "modules" / "static-web-app.bicep").read_text()
        self.assertIn("targetScope = 'subscription'", main)
        self.assertEqual(re.findall(r"^resource \w+ '([^@']+)@", main + module, re.M), ["Microsoft.Resources/resourceGroups", "Microsoft.Web/staticSites"])
        self.assertIn("stagingEnvironmentPolicy: 'Disabled'", module)


class SecretHandlingTests(unittest.TestCase):
    def test_secrets_are_masked_and_blank_ones_ignored(self):
        self.assertEqual(deploy_azure.redact("key=abc123 abc123", ["abc123", ""]), "key=*** ***")

    def test_the_token_goes_to_the_upload_in_the_environment_and_never_on_the_command_line(self):
        with mock.patch.object(deploy_azure, "stream") as stream, mock.patch.object(deploy_azure, "executable", return_value="npx"), contextlib.redirect_stdout(io.StringIO()):
            deploy_azure.upload(Path("site"), "S3CRET-TOKEN")
        command, env, secrets = stream.call_args.args
        self.assertNotIn("S3CRET-TOKEN", " ".join(command))
        self.assertEqual(env["SWA_CLI_DEPLOYMENT_TOKEN"], "S3CRET-TOKEN")
        self.assertEqual(secrets, ["S3CRET-TOKEN"])
        self.assertEqual(command[command.index("--env") + 1], "production")  # the CLI defaults to a preview environment
        self.assertIn(deploy_azure.SWA_CLI, command)

    def test_streamed_output_is_masked_and_a_failing_command_raises(self):
        shown = io.StringIO()
        with contextlib.redirect_stdout(shown):
            deploy_azure.stream([sys.executable, "-c", "print('deployment token SECRET accepted')"], dict(os.environ), ["SECRET"])
            with self.assertRaises(DeployError):
                deploy_azure.stream([sys.executable, "-c", "import sys; print('SECRET'); sys.exit(2)"], dict(os.environ), ["SECRET"])
        self.assertIn("deployment token *** accepted", shown.getvalue())
        self.assertNotIn("SECRET", shown.getvalue())

    def test_a_failed_command_reports_its_error_output_not_its_standard_output(self):
        script = "import sys; print('SECRET-ON-STDOUT'); sys.stderr.write('quota exceeded'); sys.exit(3)"
        with self.assertRaises(DeployError) as caught:
            deploy_azure.run([sys.executable, "-c", script])
        self.assertIn("quota exceeded", str(caught.exception))
        self.assertNotIn("SECRET-ON-STDOUT", str(caught.exception))

    def test_deployment_outputs_are_unwrapped_and_an_incomplete_deployment_is_rejected(self):
        self.assertEqual(deploy_azure.outputs_of(OUTPUTS)["staticWebAppUrl"], "https://demo.example")
        with self.assertRaisesRegex(DeployError, "staticWebAppUrl"):
            deploy_azure.outputs_of(json.dumps({"resourceGroupName": {"value": "x"}, "staticWebAppName": {"value": "y"}}))


class LiveSiteCheckTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.served = Path(temporary.name)
        self.write("index.html", "<h1>start</h1>")
        self.write("lab one/app.js", "console.log('a space in the name')")  # must be requested as lab%20one/app.js
        self.manifest = {"files": {
            name: {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
            for name, data in ((p.relative_to(self.served).as_posix(), p.read_bytes()) for p in sorted(self.served.rglob("*")) if p.is_file())
        }}
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(QuietHandler, directory=str(self.served)))
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        self.url = f"http://127.0.0.1:{server.server_port}"
        patch = mock.patch.dict(os.environ, {"no_proxy": "127.0.0.1", "NO_PROXY": "127.0.0.1"})
        patch.start()
        self.addCleanup(patch.stop)

    def write(self, name, text):
        path = self.served / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)

    def check(self):
        return deploy_azure.verify_site(self.url, self.manifest, workers=2, attempts=1, pause=0)

    def test_identical_files_pass_and_the_configuration_file_that_azure_keeps_back_is_not_expected(self):
        self.manifest["files"]["staticwebapp.config.json"] = {"bytes": 2, "sha256": "0" * 64}
        self.assertEqual(self.check(), [])

    def test_a_changed_file_and_a_missing_file_are_each_reported(self):
        self.write("lab one/app.js", "console.log('tampered')")
        (self.served / "index.html").unlink()
        problems = self.check()
        self.assertEqual(sorted(problem.split(":")[0] for problem in problems), ["/", "/index.html", "/lab%20one/app.js"])
        self.assertTrue(any("HTTP 404" in problem for problem in problems))
        self.assertTrue(any("content differs" in problem for problem in problems))

    def test_waiting_ends_when_the_new_start_page_is_served_and_gives_up_when_it_never_is(self):
        deploy_azure.wait_until_live(self.url, self.manifest, timeout=5, pause=0)
        self.write("index.html", "<h1>an older deployment</h1>")
        with self.assertRaisesRegex(DeployError, "did not serve the new start page"):
            deploy_azure.wait_until_live(self.url, self.manifest, timeout=0.2, pause=0.05)


class StepOrderTests(unittest.TestCase):
    def setUp(self):
        self.calls = []
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        manifest = Path(temporary.name) / "manifest.json"
        manifest.write_text(json.dumps({"files": {"index.html": {"bytes": 1, "sha256": "x"}}}))
        outputs = {"resourceGroupName": "rg-demo", "staticWebAppName": "swa-demo", "staticWebAppUrl": "https://demo.example"}

        def record(name, result=None):
            def call(*args, **kwargs):
                self.calls.append(name)
                return result
            return call

        patch = mock.patch.multiple(
            deploy_azure, MANIFEST=manifest, executable=lambda name, hint: name, signed_in=lambda subscription: {"name": "Demo", "id": "1", "user": {"name": "me"}},
            stage=record("stage"), what_if=record("what-if"), ensure_provider=record("provider"), deploy_infrastructure=record("infra", outputs),
            existing_infrastructure=record("existing", outputs), deployment_token=record("token", "tok"), upload=record("upload"),
            wait_until_live=record("live"), verify_site=record("verify", []),
        )
        patch.start()
        self.addCleanup(patch.stop)

    def run_deploy(self, *argv):
        with contextlib.redirect_stdout(io.StringIO()):
            deploy_azure.deploy(deploy_azure.parse(list(argv)))

    def test_the_site_is_staged_before_anything_is_created_and_checked_after_the_upload(self):
        self.run_deploy()
        self.assertEqual(self.calls, ["stage", "provider", "infra", "token", "upload", "live", "verify"])

    def test_a_site_that_cannot_be_staged_stops_everything_before_azure_is_touched(self):
        with mock.patch.object(deploy_azure, "stage", side_effect=DeployError("too big")), self.assertRaises(DeployError):
            self.run_deploy("--sku", "Free")
        self.assertEqual(self.calls, [])

    def test_what_if_previews_the_infrastructure_and_nothing_else(self):
        self.run_deploy("--what-if")
        self.assertEqual(self.calls, ["what-if"])

    def test_skip_infra_reuses_the_existing_site_and_skip_site_deploys_only_the_infrastructure(self):
        self.run_deploy("--skip-infra")
        self.assertEqual(self.calls, ["stage", "existing", "token", "upload", "live", "verify"])
        self.calls.clear()
        self.run_deploy("--skip-site")
        self.assertEqual(self.calls, ["provider", "infra"])

    def test_a_live_site_that_differs_from_the_staged_one_fails_the_deployment(self):
        with mock.patch.object(deploy_azure, "verify_site", return_value=["/lab/app.js: HTTP 404"]), self.assertRaisesRegex(DeployError, "/lab/app.js: HTTP 404"):
            self.run_deploy()

    def test_skip_verify_does_not_download_the_site(self):
        self.run_deploy("--skip-verify")
        self.assertEqual(self.calls, ["stage", "provider", "infra", "token", "upload"])

    def test_verify_only_checks_the_live_site_and_changes_nothing(self):
        self.run_deploy("--verify-only")
        self.assertEqual(self.calls, ["existing", "live", "verify"])


if __name__ == "__main__":
    unittest.main()
