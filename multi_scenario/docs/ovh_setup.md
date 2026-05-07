# OVH AI Training — one-time setup

Prerequisites for running experiments on OVH Public Cloud's AI Training
service. Do this **once per machine** — afterwards `multi-scenario upload-code`
and the OVH submission helpers Just Work.

The `ovhai` CLI is a **Go binary**, not a Python package — you cannot `pip install`
it. It lives outside `pyproject.toml`; install it via the OVH-published shell
script.

## 1. Install the `ovhai` CLI

```bash
curl -sSf https://cli.bhs.ai.cloud.ovh.net/install.sh | bash
ovhai --version    # expect: ovhai <X.Y.Z>
```

The binary lands at `~/bin/ovhai`. Make sure `~/bin` is on your `PATH`.

## 2. Authenticate

```bash
ovhai login
```

Opens a browser for the OVH SSO flow; the resulting token is saved at
`~/.config/ovhai/...`.

Verify:

```bash
ovhai user list      # should print your user record
```

## 3. Create object-storage buckets (one per machine, NOT per experiment)

Pick names you'll keep — they go into `configs/ovh.yaml` next.

```bash
# Pick a region: GRA / BHS / SBG / WAW.
ovhai bucket create GRA ms-code        # framework + experiment yamls live here
ovhai bucket create GRA ms-results     # per-run outputs sync back here
```

> **Why two buckets?** `bucket_code` is mounted read-only inside every job;
> `bucket_results` is mounted `:rwd` per-experiment-prefix so parallel jobs
> can't overwrite each other during OVH's FINALIZING sync (see project
> memory: trailing-slash + per-experiment-prefix gotchas).

## 4. Generate S3 credentials for `boto3`

The `ovhai` CLI authenticates jobs but doesn't help `boto3` reach the buckets.
You need a separate S3-compatible access key + secret.

```bash
ovhai user create --role objectstore-operator --description "multi_scenario S3 access"
ovhai user info <user-id> --output json | jq '{access, secret}'
```

Add the resulting credentials to `~/.aws/credentials`:

```ini
[ovh]
aws_access_key_id = <access>
aws_secret_access_key = <secret>
```

…then point boto3 at that profile **either** via `AWS_PROFILE=ovh` env var
**or** by setting these directly:

```bash
export AWS_ACCESS_KEY_ID=<access>
export AWS_SECRET_ACCESS_KEY=<secret>
export AWS_REGION=gra      # lowercase OVH region for the S3 endpoint
```

## 5. Configure `configs/ovh.yaml` + `configs/s3.yaml`

Copy `configs/ovh.yaml.example` → `configs/ovh.yaml` and fill in:

```yaml
region: GRA
flavor: ai1-1-gpu
n_gpu: 1
bucket_code: ms-code           # from step 3
bucket_results: ms-results
# default_runner template auto-substitutes {mount_code} / {yaml_path_in_container}
```

Create `configs/s3.yaml` for the code uploader:

```yaml
bucket: ms-code
prefix: ""                                       # upload at bucket root; empty prefix is fine
region: gra
endpoint_url: https://s3.gra.io.cloud.ovh.net    # OVH Object Storage S3 endpoint
```

> The `endpoint_url` is what tells `boto3` "talk to OVH Object Storage, not
> AWS S3". Match it to your bucket's region.

## 6. Verify connectivity

```bash
ovhai bucket object list ms-code@GRA      # via ovhai
aws --endpoint-url https://s3.gra.io.cloud.ovh.net s3 ls s3://ms-code/   # via boto3
```

Both should succeed (and probably show empty buckets).

## Reference

- Region codes: `GRA` (Gravelines), `BHS` (Beauharnois), `SBG` (Strasbourg), `WAW` (Warsaw).
- Available flavors: `ai1-1-gpu` (V100S 32GB), `ai1-1-cpu` (CPU smoke).
  Run `ovhai capabilities flavor list` to check current availability.
- Volume mount permissions: `:ro` (read-only), `:rw` (read-write, deletions
  NOT propagated), `:rwd` (read-write-delete, full sync). Always use `:rwd`
  for results buckets.
- See `multi-scenario` README for the per-job submission flow.
