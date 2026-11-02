# VPM release process

The public Unity package is sourced only from
`unity/Packages/com.lemolatoon.udoncl`. Compiler sources, Python tooling,
generated model bundles, and build outputs are not part of the VPM archive.

## One-time GitHub configuration

1. Create the repository variable `PACKAGE_NAME` with the value
   `com.lemolatoon.udoncl`.
2. In **Settings → Pages → Build and deployment**, select **GitHub Actions** as
   the source.
3. Keep the default Actions `GITHUB_TOKEN` enabled. No Unity license or custom
   repository secret is required for packaging or listing publication.

The listing is published at
`https://lemolatoon.github.io/udoncl/index.json`.

## Validate locally

Run the dependency-free structural validator:

```text
python scripts/validate_vpm_package.py
```

When the VPM CLI is installed, also validate the actual VPM manifest:

```text
vpm check package unity/Packages/com.lemolatoon.udoncl
```

Add the local package to a disposable Worlds project before releasing:

```text
vpm add package unity/Packages/com.lemolatoon.udoncl -p path/to/WorldProject
```

The repository `python scripts/precommit.py` gate includes the structural VPM
validation before its compiler and Unity tests.

## Publish a version

1. Update `version` in the package `package.json` using SemVer.
2. Add the same version to `CHANGELOG.md` and replace `Unreleased` with the
   release date.
3. Run the repository precommit gate and commit the release metadata.
4. Push the commit.
5. Run **Build VPM Release** from the GitHub Actions tab.

The release workflow validates the package, creates a zip whose root contains
`package.json`, and publishes the zip plus the source manifest to a GitHub
Release tagged with the version. A prerelease suffix such as `-beta.1` marks the
GitHub Release as a prerelease.

Publishing or changing a GitHub Release triggers **Build VPM Listing**, which
uses VRChat's official listing builder and deploys `Website` to GitHub Pages.

Published versions and their release archives are immutable. Do not delete an
old release or reuse its version, because existing VPM projects may depend on
that exact archive.
