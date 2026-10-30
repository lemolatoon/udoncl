# udoncl Unity Editor package

This Unity 2022.3 package imports `udoncl` compiler artifacts into stable paths under `Assets/udoncl-generated/<program-name>`. It targets the VRChat Worlds SDK 3.x (`com.vrchat.worlds`) and has no UdonSharp dependency.

## Import

Use **Tools → udoncl → Import Artifacts...** and select a compiler output directory containing `manifest.json`, its host `.uasm`, and any referenced generated `.shader` files. The importer checks schema v1, the runtime ABI and target, Worlds SDK version, names, extensions, UTF-8 content, referenced files, and relative paths before writing any assets.

It copies the manifest, `.uasm`, and shaders to stable asset paths, creates or updates one Material per device kernel, and creates an initial 256×1 linear RGBA32 particle texture plus two point-filtered, linear 256×1 RenderTextures. The particle texture stores x, y, vx, and vy as four successive little-endian float32 RGBA texels per particle. Its 64-particle state matches the native triangular LJ lattice and its zero-momentum initial velocities.

On import failure, the importer restores prior file contents and `.meta` files and removes newly created assets. Existing asset paths retain their GUIDs. Shader compilation is checked by loading each imported shader; a missing shader aborts and rolls back the import.

## Udon Assembly and prefab

The package deliberately leaves `.uasm` to the Worlds SDK's public
`UdonAssemblyProgramAssetImporter`. It calls `RefreshProgram`, checks
`AssemblyError`, retrieves the serialized program, then binds it to a generated
`LennardJonesDemo.prefab` through the public `UdonBehaviour` and variable-table
APIs. It does not register a competing `.uasm` importer and uses no private
reflection or UdonSharp.

The prefab contains an interactive display quad. The Udon host executes the two
generated fragment kernels every `_update`; interaction injects heat and the
amplitude decays while the kernel's Langevin term cools the system back toward
the crystal. Drop the prefab into a World SDK scene to run it.
