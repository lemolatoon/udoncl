# PLAN.md — VRChat向け C / OpenCL C → Udon / HLSL コンパイラ

> Status: **Udon/Native Async, Host storage allocation, compiler-generated U#,
> and Unity MNIST numerical gate implemented; CUDA-shaped asynchronous Device
> library migration is next**

The current read/write resource-alias contract, `cl_mem` Device API migration,
and MNIST acceptance work are specified separately in
[`DEVICE_RESOURCE_ALIAS_PLAN.md`](DEVICE_RESOURCE_ALIAS_PLAN.md). That plan is
the authoritative implementation order for this checkpoint; this file keeps
the longer-term architecture and historical checkpoints.
> Design cut-off: **2026-11-05**
> Implementation amendments (user): **LLVM/MLIR 23 baseline; Windows host builds supported by design.**
> Initial toolchain pin: **LLVM/MLIR/Clang 23.1.4** (see `cmake/llvm-lock.json`).  
> Audience: Codex / compiler・runtime・Unity integration 実装担当  
> Target: VRChat Worlds **PC (Windows)** first; macOS/Linux are host-side compiler development environments.  
> Source of truth: この文書。独自Dialectの**全Operation（初期設計の閉じた集合）**を§6–§9に列挙する。

## 0. 依頼・完成条件・設計上の区別

**目標**: Cで書かれた計算ライブラリをLLVM/MLIRを通してUdon AssemblyへAOTコンパイルし、**UdonSharp (U#) がエントリーポイント**としてそのライブラリを呼び出す。C側からOpenCL 1.2互換サブセットAPIでOpenCL C Kernelを使い、Device CompilerがOpenCL CをFragment Shader (ShaderLab/HLSL) にAOT変換する。VRChatワールド内では`VRCGraphics.Blit`、RenderTexture、`VRCAsyncGPUReadback`等を利用してGPU計算を実行する。GPU待機中のC呼び出しは中断・再開できる。

**初期End-to-End/MVP**: U# → C export関数 → OpenCL Buffer/Kernel → GPU実行 → 非同期Readback → C関数再開 → U#完了通知。SAXPYに加えて、**`__local` + `barrier`を用いたfloat32 Sum Reduction（複数work-group、最終1値まで）**を実行して検証する。Picolibcを中心とした**ISO C11 libc全体への対応は最終目標**とし、MVPと同時に完全対応済みだとは主張しない。

**区別すべき3種類の事実**:
1. **確定した設計**: 本文のABI/機能分担/Operation catalog。実装を指示する仕様。
2. **現行仕様に根拠がある事項**: VRChat Udon VMの9命令、UdonBehaviour間呼び出し、VRChat GPU APIs等。参照先は§20。
3. **技術的仮説・実機検証待ち**: Fence Blit + Readbackが先行GPU work全体の完了を証明するか、Udonで高速なTexture uploadができるか、Program AssetをPublic SDK APIだけで自動生成できるか等。§17を通すまで「動作保証」と書かない。

**範囲外（初期）**: Quest、OpenCL 2.x/SVM、OpenCL Image API、任意scatter/atomics、GPU `double`/64-bit整数、Host `pthread`/POSIX process/network、複数Host Execution Context・複数Command Queue、ランタイムOpenCL Cコンパイル、VRChatネットワーク同期、Device再帰、OpenCL完全準拠の宣言。

## 1. End-to-Endアーキテクチャ

```text
User-written UdonSharp (entrypoint, app orchestration, optional libc service)
  └─ compiler-generated low-level typed U# export binding
       └─ generated C host UdonBehaviour (per-instance globals / heap / frames)
            └─ C OpenCL runtime (Picolibc-compatible C + __vrc_* intrinsics)
                 └─ Udon/VRChat GPU APIs → GPU RenderTextures → AsyncGPUReadback

Host .c / libc sources                     Device .cl sources
   │                                          │
 Clang (C11 target / ILP32)                  Clang OpenCL C 1.2 (SPIR-aware)
   │ LLVM IR + metadata                       │ LLVM IR + kernel metadata
   └─────────────┬─────────────────────────────┘
                 ▼
          MLIR LLVM Dialect importer
                 ▼
     LLVM IR → High-level MLIR reconstruction
      (independent CMake library, reused both sides)
         ┌───────┴────────┐
      Host path         Device path
         │                 │
   func/cf/scf/arith     gpu/scf/memref/affine/arith
   + udonhl + vrc       + OpenCL semantic & dependence analyses
         │                 │
   MLIR Async +        barrier elimination / cost-aware phase splitting
   backend coroutine      │
         │              vrcgpu (fragment-safe passes)
      udonasm              │
         │             direct HLSL + ShaderLab
      .uasm                │
         └───────────┬─────┘
                     ▼
        manifest.json + binding.cs + debug/dumps
                     ▼
    Unity Editor Importer (UPM package)
    - Udon Program Asset / Behaviour binding
    - Shader / Material / Texture configs
    - compiler-produced U# binding import / compilation
    - asset GUID preservation + ABI validation
                     ▼
         VRChat PC World (Build & Test)
```

**Compiler/Unity責務**: CLIは`.uasm`, `.shader`, `manifest.json`, compiler-generated `binding.cs`, diagnostics/debug dumpsを自己完結artifactとして生成する。physical export adapter、Manifest、低水準U# bindingは同じMLIR export ABIから生成する。Editor Importerは`.asset`、Material、C# import/compile、参照接続、GUID保持を担当し、binding sourceを再生成しない。高水準ONNX Session U# wrapperは初期scope外とする。

**MLIRパイプラインにおける共存**: `func`, `arith`, `math`, `scf`, `cf`, `memref`, `gpu`, `affine`, `llvm`など標準Dialectを積極的に使う。`vrc`, `udonhl`, `udonasm`, `vrcgpu`は固有意味論に必要な操作だけを定義する。以下のOperation名・署名は**これから実装する独自仕様**であり、既存MLIR本体やVRChatに実装済みという意味ではない。

## 2. 決定事項のサマリー（Grilling 1〜134）

### 2.1 GPUメモリ／Kernel

- Device論理アドレスは**統一された32-bit byte-addressable空間**。`cl_mem`は別の32-bit opaque handleで、device addressではない。64-bit ABIは将来拡張。
- 論理的には統一空間でも、物理実装は**複数Texture**。初期は原則1 `cl_mem` allocation → 1物理Texture。将来page分割できる間接マップを設計する。
- Buffer layoutは意味論的にはraw byte。初期汎用形式はRGBA8（sRGBを無効にした明示的なbyte encoding）。**静的に安全と証明できるallocation**のみRFloatなどのtyped textureに最適化。混合access/`ptr` cast/aliasの際は**allocation全体でraw fallback**。複数Kernelを横断して解析する。
- 1D論理配列を2D Textureへmappingする**ABIを規定**する（§11）。Texture size/offset/format/strideはManifest・Runtimeから供給する。部分更新の未書込領域を保持し、同じTextureをBlitのsourceとdestinationとして使わず**ping-pong**する。
- Host→GPUはtyped Udon array → staging Textureを第一候補。**Q18は性能検証を条件に保留**。部分転送`offset/size`は初期から対応。GPU→Hostは`VRCAsyncGPUReadback`で全Texture読み込み→範囲コピーを初期実装とする。部分Readbackの最適化は後回し。
- Kernelごとの固定Texture/Uniform Binding上限（実測に基づくprofile。未検証の固定数を捏造しない）。動的な任意allocation selectionは初期エラー。
- Buffer解放は未完了commandがある間延期（deferred reclamation）。GPU allocationはTexture単位、論理メモリはFree Listで再利用可。
- GPUはOpenCL C 1.2サブセット：1D/2D、3Dはlinearized、実行時global size/offset、barrierを含むlocal sizeはコンパイル時特殊化、補助関数はinline、再帰不可。初期Device数値型は8/16/32-bit intとf32、64-bit int/f64は後日。Image/Sampler APIは後日。
- Fragment-safe single-passを優先し、複数出力・Reduction・Scan等は選択的な**cost-aware multipass**。Barrier elimination + cost-aware phase splitting、`__local` scratchを中間Textureに置換。divergent barrierや解析困難な依存関係はコンパイルエラー。一般scatter/atomicsは証明済み変換がなければ拒否。MRTはMVP非依存、**1 pass = 1 RenderTexture output**。
- Reduction MVPはwork-group内+group間の完全なfloat32 sum。Tree Reductionの中間値はGPU上で維持し、最後だけReadback。数値検証は許容誤差（絶対/相対）、再結合によるbit差を許容。
- static Cost Report：pass数・Texture read/write bytes・scratch・警告・変換可否。高コストな変換に`--allow-expensive`。

### 2.2 Host、OpenCL Runtime、libc

- Hostは**32-bit ILP32, little-endian**を固定ターゲットABIとする。Udonの内部値型への変換・byte memory・構造体alignmentは明示。GPU ABIとは別。
- Host C pointerは32-bit flat byte virtual addressとし、Cのpointer identity、alias、構造体、union、byte layoutを保つ。ただし物理storageはpointer ABIそのものではない。全Host moduleをlinkして公開export以外を内部化し、SROA、mem2reg、inlining、定数伝播、descriptor specialization、DCEを行った後でstorageを決める。
- addressを観測されないSSA scalar、引数、戻り値、一時値、local、global scalarは、少数の型付きscalar Udon Heap slotを物理レジスタとしてinterprocedural coloringする。収まらない値は物理Udon型ごとの固定長typed arrayへspillする。固定公開ABI slotとUnity resource slotはallocator外とし、入口/出口だけregisterとcopyする。
- pointer identity、dynamic indexing、byte alias、`memcpy`等が最適化後も残るobjectだけをraw linear memoryへ置く。raw memoryはlittle-endianのword-packed `SystemInt32Array`（1 element = 4 logical bytes）とし、aligned `i32`/`f32`を1回のarray accessへ、byte/unaligned accessをshift/mask/read-modify-writeへlowerする。現在の1 element = 1 logical byte方式は移行実装であり最終方式にしない。
- 初期Host backendは非再帰call graphだけを許可し、再帰SCCを明示errorにする。将来の再帰対応は型別spill array内のdynamic frameまたはUdonSharp `[RecursiveMethod]`相当のsave/restoreとして別段階で設計し、非再帰の静的coloringへ混ぜない。`udonasm`に擬似CALLは追加しない。
- 1 library instance = 1 Host Execution Context; 1 in-order OpenCL Command Queue。複数pending GPU requestsを管理できる構造にするがrequest識別が不確実ならMVPは直列化。別C呼び出しがbusy中はBusy拒否。完了callbackはbusy解除後にU#へ通知し、再呼び出し可。
- `cl_mem`は32-bit handle table。`clSetKernelArg`の引数は**設定時にコピー**。OpenCL 1.2 AOT互換サブセット。`clCreateProgramWithSource`は静的に既知のKernel Manifestを解決し、`clBuildProgram`は既存Assetと対応付けるだけ。実行時の任意OpenCL C source生成は未対応。
- OpenCL APIの契約・エラー処理は**C Runtime**。Udon/VRChat GPU操作・Service呼出・同期は`__vrc_*` compiler intrinsics → `vrc`へ。Host + Runtime LLVM IRを静的リンク。
- libcは**ISO C11標準ライブラリ全体を対応目標**とし、Picolibcを第一候補として移植。環境依存部分（console/VFS/time等）は必要ならU# Serviceで実装し、Cと接続。初期VFSはinstanceごとのin-memory、U#サービス側への委譲も可。stdin bufferにデータがなければ初期はEOF、stdinでの非同期待機は後日。stdout/stderrはC側buffer→一括U#通知。
- `malloc`/`free`は容量上限・失敗時`NULL`、ライフサイクルは`Initialize/Reset/Dispose`。環境依存C関数に実装不可能な処理があれば静かに偽装せず、ドキュメント化し、適切なエラー/制約を返す。
- Safepointにより長いCPUループを複数フレームに分割（最初は明示yield、以後自動Budget）。GPUの非同期callbackと同様にResume stateが生存する。

### 2.3 U# API、Unity、デバッグ

- **アプリのエントリーポイントはユーザーのUdonSharp**。`main()`自動実行ではない。Cはリンク可能なHost libraryであり、`UDONCL_EXPORT`を付けた定義だけを公開する。注釈がexport発見の正本で、schema-v2 configは公開名・parameter表示名・binding namespace/class・callback契約などをoverrideする。
- Exportごとのphysical entry adapter、Manifest、**低水準型付きU# binding**を同じMLIR export ABIからcompilerが生成する。同期呼出は予約Heap変数へのSetProgramVariable →公開Custom Event → GetProgramVariable、非同期はStart +完了通知Event/Statusを用いる。Editorはcompiler artifact内の`binding.cs`をimport/compileするだけで再生成しない。
- 同期/非同期はCall Graph解析で自動判定し、GPU待機可能性が呼出元まで伝播する。C/OpenCL semanticsを保ったままMLIR Async dialectへ変換し、上流`async-func-to-async-runtime`にcoroutine CFGを生成させる。U#には`async/await`を要求しない。
- `char *`は通常のC byte pointerのまま維持し、U# stringへ暗黙変換しない。U# stringは明示的な`udon_string_t` immutable UTF-16 referenceとして公開し、C Hostはlength/code-unit/scalar accessorで全体コピーなしに読む。基本型・Udon typed arraysは同一Udon objectをcaller-owned storageとして直接参照し、構造体はWrapperマーシャリング、生ポインタはpublic APIに出さない。
- 1 UdonBehaviour instanceごとにGlobals/Heap/Stack/VFS/サービスBindingが独立。`Initialize()`, `Reset()`, `Dispose()`とgeneration counterで古いGPU callbackを無視。未完了commandのTextureは安全になるまで保持。
- 非同期失敗も完了Eventで通知＋Status（成功と区別）。Debug watchdogは診断のみで誤判定を避ける。
- Unity統合は**外部Clang/MLIR CLI + Unity Editor Importer**、UPM package。Compilerがbinding C#まで生成し、ImporterはShader/Material/Udon Program Asset生成、C# import、script reload後のBindingを行う。GUID保持・トランザクション的Import（失敗時既存正常Asset維持）。ManifestにCompiler/Runtime/SDK/ABI versionを記録し検証する。
- Udon ExternデータベースはインストールSDKの公開情報から生成/検証。`udonasm`は実opcode 1:1、TableGenで4 Dialectのop/type/attrを定義し、各々独立CMake library。**LLVM IR→High-level MLIR復元はそれ自体が独立CMake library**。
- 開発時は、利用可能な**SDK付属Udon VM**を第一に利用。Unity Test Runner / ClientSim、実VRChat Build & Test。スタンドアロンVMが実証できれば活用するが、独自VMやFastUdonVMのみで正しさを断言しない。Compiler自体はmacOS/Linux上でUnityなしでもbuild/test可。

### 2.4 U# Host API・Binding実装の確定事項（Grilling 135以降）

- 対象を当面 Unity 2022.3.22f1 / VRChat Worlds SDK 3.10.5 に固定する。論理export ID・公開名・TypeSchemaはUdonCLの安定ABI、event/parameter/result slotは当該UdonSharp compilerに合わせたversioned physical ABIとする。
- ユーザー向け入口はcompiler生成の低水準型付きU# export bindingであり、raw event/slot ABIは内部実装とする。公開APIの形はユーザーがC Hostのexportとして設計する。高水準ONNX Session U# wrapperは後続機能とする。
- U#からCへ渡すprimitive arrayは安全性と寿命を証明できる場合に同一Udon arrayを借用してzero-copyとする。Unity Device memoryは明示的な固定allocationと全量CopyBufferを検証済み。次は一般Host MLIR/continuationと公開UdonBLAS SGEMM Unity gateを最優先し、その後ONNX MLIRへ進む。Map/SubBufferとhost-backed bufferは後回しとする。
- 公開handleは一意なUdonSharpBehaviour instanceとし、物理GPU resourceは内部poolで再利用する。handleのgeneration/lifetime検査と物理resource所有権を分離する。
- Compiler生成bindingは同期/非同期exportを同じMLIR export ABIから扱い、`binding.cs`、binding path/namespace/class、ABI fingerprintを生成する。設定の`public_name`は`Try`/`Start`を含まない論理名とし、生成規則から同期`TryX`・非同期`StartX`・status/result accessorを一意に作る。Manifestの`element_type`/`public_access`からdtypeどおりのwhole-buffer `TryWrite<Name>(float[]|int[])` / callback-first `StartRead<Name>(float[]|int[], receiver, event)`もcompilerが生成する。利用者readbackとC Host内部command readbackは一つのcallbackでmode分岐し、generationと`busy`/`readbackBusy`で寿命・再入を検査する。公開buffer APIを決めるID・名前・論理byte長・要素型・入出力accessもcanonical ABI fingerprintへ含め、Importerが独立に再計算する。Importerはcompiler sourceを配置するだけで再生成しない。旧Editor generatorはUnity removal gateまで互換経路としてだけ保持する。初期対応型は`int`、`float`、`int[]`、`float[]`とcallbackに必要なUdon型で、未対応型はboxingせずcompile errorにする。
- 現行の一般ONNX Unity bundle、`UdonOnnxRuntimeBehaviour`、`UdonOnnxModelAsset`、texture arena、`static_inputs`、Material直接schedule、execution-plan JSONは、Arena AddとTiny LlamaをEditor/D3D11で実行できた**旧経路の検証資産**である。最終設計の実行基盤とはせず、新Host MLIR経路でMNISTが合格した時点で関連generator/parser/runtime/importerを一括削除する。
- 現行Llama 3.1 8Bのplan-only preflight（546 node、225 Q4 call、322 direct Device pass、64 view、232 atlas、4,690,411,520 bytes）とexternal weight検査は容量・入力検証の事実として保持するが、execution planを新経路へ持ち込まない。ライセンス済み実重みがないため、全weight artifact、Unity import、数値推論は未検証のままである。
- 現行Arena Add/Tiny Llama package sampleとscene配置UIは移行回帰用として、新Host MLIR経路の同等artifactが成立するまで保持する。移行後はexecution-plan依存を除去し、共通generated U# bindingだけを使用する。
- packaged C/OpenCL Hello DispatchとSAXPYはimport時にgenerated typed bindingだけでなく、その公開`Try*`/named Write/async Read APIだけを使うU# controllerも生成する。domain reload後にcontrollerをbinding prefabへ自動添付し、binding参照を設定することで、配置したprefabをInteractしてend-to-end実行できる。raw event/heap slotはsampleへ露出しない。
- 同じExamples menuは標準OpenCL Host APIのhello dispatchとSAXPY artifactも同梱し、compiler checkoutなしのimportからgenerated U# binding/prefab完成まで試せる。source CMake workflowは編集・再生成経路として維持する。
- 固定Device bufferは`element_type`と`public_access`をOpenCLの実型・read/write flagに照合し、compilerが物理buffer IDを隠した`TryWrite<InputName>` / `StartRead<OutputName>` U# APIを生成する。Importerは生成sourceとfingerprintを検証・配置する。buffer ID指定APIは低レベル互換面として残す。
- C/OpenCL artifactのimport時にscene配置を選択でき、compiler-generated low-level U# bindingのcompile/domain reload後にprogram-scopedな一回限りのrequestを消費してbinding prefabのlinked instanceをactive sceneへ配置する。
- `vrc.await`/event境界からmay-suspend call graphを固定点解析し、helperをinlineせず`async.func`/`async.call`/`async.await`へ変換した後、上流`async-func-to-async-runtime`まで通すC++ passを実装済み。Nativeでは同じ共通Async IRをLLVM coroutine/runtimeへlowerし、外部完了による親子call、同期点を挟んだ2回の連続suspend/resume、呼出し前に完了済みの即時経路を`lli`で実行して結果を検証済み。実UdonBLAS SGEMM call graphでも親子coroutineを分離した`host.async-runtime.mlir`をartifactへ出力できる。対応するRuntime/Coro op集合を検証し、symbol順のstatic frame IDと各suspendのresume PCを付与するUdon準備passも接続済みで、同artifactでは10 frame・17 suspendを計画できる。MLIR livenessによるtyped slot coloringと、PC/spill/async-object state・waiter・payloadのcompiler-private `udonhl.global` materialization、suspend前のPC/live-value保存、resume block先頭の型付き復元も実装済み。復元値はdominanceに基づいて各useへ接続するため分岐CFGを直列化せず、`i64`も8-byte slotとして保持する。direct call/returnを横断してcalleeのreturn objectとcallerのcall resultを同一canonical objectへ統合し、その領域へpending reset、waiter frame/PC、payload store/load、available/error state書込みを接続した。state encodingはpending=0/available=1/error=-1に固定し、ready/error判定をUdonHL load/compareへ変換してready済みobjectはsuspendを迂回する。最終Udon Async passは残る`async.runtime.*`/`async.coro.*`を除去し、event objectを`udonhl.async_bind_event`へ変換し、全resume requestを1個の`__udoncl_async_resume_dispatch`へ集約する。scalar UdonASM loweringはその17個の`udonhl.async_enter`をresume blockへの直接entryと静的return slotへ変換する。外部event用にはcanonical object ID公開slotとcompiler-private `__udoncl_async_complete(object_id,status)`も生成し、payload/state更新、waiter clear、dispatcher呼出しまで接続済み。SGEMMの最終IRは10 frame、17 resume point、127組のspill/reload、12 canonical object、17 waiter登録、22 dispatcher call（外部completion 2 branchを含む）、17 static entry、2 event bindingを持ち、動的`udonhl.async_resume`、Async Runtime/Coro、`vrc.event_as_async`の残存は0である。再帰、`async.execute`、may-suspend関数内のindirect callは明示errorにする。この最終経路はproduction assemblyへ接続済みであり、残るgateはUnity MCP上のcompiler-generated callback pump検証である。
- 既存の`udonhl.frame_*`/`suspend`による単一await直接変換は、過去の実行検証を保持する移行用production経路であり、機能拡張先ではない。新Async backendがcompiler/Native/Unity/MNIST gateを通過した時点で、forced inline、手動effect propagation、Editor binding generatorと一緒に削除する。
- CMakeは通常の複数source/target記述を入力とし、preset対応は後回しとする。Windows向けcompiler/toolchainはruntime packageと分離したVPM packageとして配布し、将来は`udonclc`を同梱してUnity UIからCMake directoryを指定できるようにする。
- CMake-toolchain MVPは`find_package(UdonCL CONFIG)`とHost/Device source-library target、artifact targetとして実装済み。各targetの複数translation unitをClangでLLVM bitcode化し、target固有usage requirementsを保ったまま`llvm-link`でHost/Deviceごとに結合してから`udonclc`へ渡す。外部consumer CTestが各3 source・各2 targetと生成bundleを検証する。preset、Unity UI、compiler VPMへの実バイナリ同梱は後続。
- `udoncl_add_unity_library`も`DEVICE_LIBRARIES`を受け取り、複数Device targetをlinkして選択したAOT kernelのShaderLab/HLSLとmanifestをUnity artifactへ同梱する。public CMake→Importer経路では、標準OpenCL Host APIで書いた固定AOT dispatchをtyped dispatchへ回復し、ordered read-only input buffer群 + 1 write-only output buffer + copied f32/i32 scalar群を生成materialへ束縛できる。`examples/saxpy`が2 buffer + i32/f32 scalarを通す。複数kernel、任意argument順、dynamic NDRangeは後続。
- typed dispatch backendは`vrc.dispatch_desc_create`で`clSetKernelArg`時にcopyされるf32/i32 scalarを保持し、複数bufferを持つ`vrc.dispatch`をinput texture/scalar property設定とoutputへの`VRCGraphics.Blit`へloweringする。Importer finalizerはmanifestのdispatch siteから生成materialを公開Udon heap slotへ束縛する。標準OpenCL Host bridgeは直線的なsetup/build/setarg/enqueue/release列をconfigと照合してこのtyped IRを生成し、buffer/scalar順、read/write flag、context/queue/program/kernel/argument/releaseの由来が異なるcall列を明示的に拒否する。
- compiler packageにはpin済みKhronos OpenCL Headersとそのlicenseを同梱し、`udoncl_add_host_library`から`<CL/cl.h>`を追加設定なしで使えるようにする。Unity向けには、1 platform/GPU device、1 context/in-order queue、constant-size `clCreateBuffer`、全量`clEnqueueCopyBuffer`、またはbuffer群を先頭に並べた固定AOT gather dispatch、releaseからなる直線的なOpenCL 1.2 subsetをtyped IRへ回復する。dispatchはread-only input群、write-only output 1個、copied f32/i32 scalar群に限定し、copy/dispatch両経路でcontext/queue/program/kernel/opaque `cl_mem`とscalar slotの由来を検証する。status依存control flow、upload/readback、任意argument順・dynamic NDRange、partial copy、一般lifetimeは未対応として明示的に拒否する。
- Device buffer C runtimeは明示的`clCreateBuffer` allocationと範囲検査付き`clEnqueueWriteBuffer` / `clEnqueueReadBuffer` / `clEnqueueCopyBuffer`を実装済み。host pointer flags、SubBuffer、Map/Unmap、矩形Copyは暗黙fallbackせず明示的に拒否する。handle/context/generation、未初期化range、overlap、event依存をnative fragment実行テストで検証する。Unity側の物理RenderTexture poolと非同期Readbackは次段階。
- Compiler IRでも最初のDevice memory契約は`vrc.buffer_alloc`と`vrc.buffer_copy`に限定し、rgba8 raw storage、明示byte size/offset、typed buffer/eventとして表す。C runtime intrinsicからこの二つのOpへの置換と定数範囲Verifierを先に実装し、Unity RenderTexture allocation/copy lowering、upload/readback、awaitは後続段階として混同しない。
- Unity library configの固定buffer宣言はcompilerがcanonicalなlogical byte size/texture shape/heap slotへ変換し、Importerがlinear RGBA8・Point・ClampのRenderTexture assetを生成、binding prefabのraw UdonBehaviourへ公開slotで束縛する。これは`vrc.buffer_alloc`の物理backing poolであり、handle table/copy command loweringが接続されるまではbuffer command実行完了とは扱わない。
- 初期Unity copy loweringは、設定順の固定poolへconstant-size `vrc.buffer_alloc`を束縛し、別buffer・offset 0・同一logical size/shapeの全量`vrc.buffer_copy`だけをraw-load copy shader + `VRCGraphics.Blit`へ変換する。copy materialもartifactから生成してUdon heapへ束縛する。部分/dynamic/self copy、release/reuse、C OpenCL runtimeからtyped intrinsicまでの接続は未実装として明示的に拒否し、byte-precise一般CopyBuffer完了とは扱わない。
- typed Host MLIR検証入口から生成したcopy artifactはUnity 2022.3.22f1 / Worlds SDK 3.10.5でimport/reimportし、固定pool/materialのPrefab binding、SDK IUdonVM `CopyFull`、RGBA8 `[19,87,155,223]`のGPU readback完全一致まで確認済み。これはClient実機検証ではなく、通常C `clCreateBuffer`/`clEnqueueCopyBuffer`からtyped intrinsicを作るruntime bridgeの完了も意味しない。
- 固定poolの最初のC bridgeとして実装した`<udoncl/unity_buffer.h>`は互換入力として残すが、公開exampleは標準OpenCL 1.2 Host APIの制限付きAOT subsetを使用する。ILP32 opaque handleはcompiler内でcontext/queue由来とともに検証し、各allocationから一つ以上のcopyへ直接流れる場合だけtyped `vrc.buffer`へ回復する。source→intermediate→destinationの2段chainをUnity import/reimport後にSDK IUdonVMで実行し、両回とも各出力8192×2全16,384 pixels / 65,536 physical bytesがmismatch 0、console最終error/warning 0、cleanup完了を確認済み。
- 公開`examples/saxpy`は通常のOpenCL Host Cからread-only input + write-only output + copied i32/f32 scalarをtyped dispatchへ回復し、Unity import/reimport後にSDK IUdonVMで`RunSaxpy(16, 2.0)`を実行済み。入力32 wordsのroundtrip、先頭16 wordsの`2*x+y`、末尾16 wordsのpassthroughはいずれもmismatch 0 / max error 0、material/heap binding一致、console error/warning 0、cleanup完了。これは固定同期AOT subsetの証拠であり、公開upload/readback・async continuation・VRChat client実行は未達。
- CMake/C source生成artifactのcopy shaderは補間UVではなくfragment `SV_POSITION`の整数texel座標を使う。8192×2 RGBA8へx/y別patternを入れ、Unity import/reimport後にSDK IUdonVMで全16,384 pixels / 65,536 physical bytesをcopyしてpixel/byte mismatch 0を確認済み（Unity 2022.3.22f1、Worlds SDK 3.10.5、Client実機ではない）。
- 公開CMake target `hello-dispatch-unity-library`は標準OpenCL 1.2 Host Cの直線的なsetup/build/setarg/enqueue/release列を、固定1-output buffer + copied f32 + 1 work-itemのtyped dispatchへ回復する。Unity 2022.3.22f1でartifactをimport/reimportし、両回ともshader supported / 1 pass、material・output RT・VM heapの参照一致、SDK IUdonVM `DispatchSquare(3.0f)` Interpret/status 0、readback `[0,0,32,65]` = exactly `10.0f`、最終console error/warning 0、cleanup完了を確認済み（Client実機ではない）。同じ経路の`hello-int-dispatch-unity-library`もimport/reimport両回でinteger property/material/RT/VM heap参照一致、SDK IUdonVM `DispatchSquareInt(3)` Interpret/status 0、readback `[10,0,0,0]` = little-endian i32 `10`、最終console 0、cleanup完了を確認済み。
- ONNXは同じ基盤上の別AOT packageとし、生成C Host orchestrationからUdonBLAS/UdonDNN/UdonATenを呼ぶ。PyTorch operatorを追加する前に、実際のPyTorch/CUDA実装が属するcomponentとinterfaceを確認して合わせる。
- U# physical ABI、scalar C→UASM、typed wrapper/Importer、初期CMake/toolchain、固定Device buffer、straight-line GPU dispatchは段階的に実装済みである。以後の順序は§2.5およびP3.5/P6を正本とする。各段階は単独でtest可能にし、commit前は`python scripts/precommit.py`だけを主agentが実行する。
- 公開CMake packageは複数Host/Device sourceを通常のCMake targetとして扱い、`udoncl_add_unity_library`で自己完結Importer入力を生成する。binding指定artifactは`manifest.json`、`host.uasm`、shader/resourcesに加えて同期/非同期対応のcompiler生成`binding.cs`を含む。Async productionも静的frame/trampoline backendを使用する。

### 2.5 Host MLIR・UdonBLAS・ONNX移行の確定事項（Grilling 2026-11-03）

#### 正規Host compiler経路

- C Hostの正規経路は`Clang → LLVM IR → MLIR LLVM Dialect → func/cf/scf/memref/arith → udonhl/vrc → udonasm`とする。`vrccl.cpp`のraw LLVM上straight-line OpenCL call matcherは互換用legacy fast pathに隔離し、新経路失敗時の暗黙fallbackには使わない。Hello、SAXPY、CopyBuffer、UdonBLAS SGEMMが新経路でNative/Unityとも合格した時点で削除する。
- 既存の`vrc-recover-high-level`が実装済みのLLVM branch/PHIから`cf`、`lift-cf-to-scf`による`scf.if`/`scf.while`回復を再利用する。新規CFG recoveryを作り直さず、現状single-blockに限定される`HighLevelToUdonHL`以降をmulti-block/SCF対応へ拡張する。公開exportから到達可能なcall graphはUdonHLへ入る前に未合法化LLVM opを残さない。
- CompilerはUdonBLAS固有patternを持たない。direct non-recursive call graph、internal/private global scalar・固定長配列、fixed local配列、typed pointer、GEP、証明済み範囲のmemory operationを一般C機能として回復する。型・範囲・aliasを証明できる配列はtyped memory、byte pointer/再解釈はraw memoryとする。未対応意味論は元location付きerrorにする。
- OpenCL APIのhandle、queue、event、status、範囲検査、argument copyは共有C runtimeが所有する。Compilerが`cl*`関数名を直接GPU opへ置換せず、runtime内部の`__vrc_*`境界だけを`vrc.*`へsubstituteする。NativeとUnityは同じC runtime契約と非同期状態遷移を使用する。

#### OpenCL非同期契約とMLIR Async lowering

- `clEnqueueWriteBuffer`/`ReadBuffer`の`CL_TRUE`と`CL_FALSE`を区別する。Non-blocking commandはenqueue後にCへ戻り、Host pointerの有効性・未変更はOpenCL契約どおりプログラマ責任とする。Blocking write/read、`clWaitForEvents`、`clFinish`だけをsource固有`vrc.await`へ変換する。
- `vrc.await`へ到達する関数をC++ MLIR passで固定点解析し、affected function/call/return/waitを`async.func`/`async.call`/`async.await`へ変換する。通常のC callはcallee完了を直後にawaitし、最適化が意味保持を証明した場合だけ移動・削除する。helperを強制inlineしない。
- Canonicalizeと必要なSCF→CFの後、上流`async-func-to-async-runtime`へ渡す。ここまでをNative/Udonで共有し、Nativeは標準Async Runtime + LLVM coroutine、Udonは生成されたAsync Runtime/Coro CFGを静的Heap frameとresume trampolineへlegalizeする。
- Udonにthread pool、CPU task scheduler、polling、time slicing、fairness policy、汎用ready queueを作らない。GPU/event completionが一意なcontinuationを選び、generated switchでresumeする。child完了は同じtrampolineでparent continuationへ進む。複数event待ちはcounterの最後の完了だけがcontinuationを起こす。
- 初期Udon backendは1 component instanceにつき1 active root、pending async objectにつき1 waiter、structured direct call、非再帰call graphだけを許可する。`async.execute`、並行async function、escaping coroutine result、indirect async call、複数waiterは明示errorにする。GPU command/event自体の並行性は保持する。
- Suspendをまたぐlive SSA値、PC、async object state/waiter/payloadを静的logical frameへ保存し、callee resultとcaller call resultを同一canonical objectへ統合するAsync runtime legalizationは実装済みである。現行実装はこれらを個別compiler-private `udonhl.global`へmaterializeするが、最終storageでは§2.7の物理型別persistent spill arrayへ集約する。liveness、resume PC、ready/error fast path、共有dispatcher、外部completion entryの意味論は維持し、保存先だけを新allocatorへ置換する。
- 通常のOpenCL/runtime failureは`cl_int` payloadとして伝播し、Async infrastructure errorと区別する。上流が生成するrefcount/lifetime opはwhole-program ownership解析で静的解決し、証明できない形は初期backendで拒否する。
- `cl_mem`は固定物理poolとは独立したgeneration付き論理handleとする。Buffer capacityはcompile/import時に事前確保し、runtime dimensionが容量を超えれば明示statusを返す。部分ReadBufferを意味論どおり扱い、初期Unity backendが全Textureをreadbackして要求範囲だけcopyすることは許可する。

#### Manifest、CMake、U# binding

- C exportは`UDONCL_EXPORT` source annotationからLLVM metadata、MLIR attributeへ伝播する。Annotationがexport発見の唯一の正本であり、schema-v2 `unity-library.json`はpublic name、parameter表示名、binding namespace/class、callback、package policyだけをoverrideする。Stable export IDはpublic signatureからcompilerが生成し、衝突はerrorにする。Async性はcompilerが推論する。
- Physical export adapter、Manifest、低水準U# `binding.cs`を同一MLIR export ABIから生成する。Manifest fingerprintはexport ID/name/order、parameter name/type、result、may-suspend effect、callback、binding namespace/class、runtime/physical ABIを含む。Compiler artifactはUASM/shader/manifest/binding source/resourcesだけでUnity import可能にする。
- Importerはcompiler bindingを受け入れる前にtyped exportとbinding metadataからcanonical v2 ABIを再構築し、SHA-256 fingerprintを再計算して完全一致を要求する。形式だけの検査にはしない。physical ABIも固定値と完全一致を要求する。
- Suspendしないexportは型付き`TryX(..., out result)`、suspend可能exportはreceiverとcompletion event名を受けるcallback-firstの`StartX(...)`とstatus/result accessorに分ける。これはU#がblocking waitできないための共通transportであり、model固有Session/request APIではない。公開Host pointerは対応U# arrayへ写し、extentは明示C引数とする。`cl_mem`はopaque handleのままでarrayにしない。
- Editor Importerはcompiler生成`binding.cs`を配置・compile・接続し、Manifestからsourceを再生成しない。新規公開APIの正本はschema v2とし、schema v1のbinding指定は既存example移行用の明示的な互換入力としてだけ保持する。
- Installed CMake packageは通常のtarget graphを使う。利用者は`find_package(UdonBLAS CONFIG)`後に`UdonBLAS::UdonBLAS`をlinkし、Host source、header、全Device kernel、runtime依存を推移的に得る。UdonCL内部metadataはHost/Device source種別を保持する。Acceptanceはsource tree hintでなく一時install prefixだけから独立projectをconfigure/buildする。
- Editor Pluginの現段階の入口はONNX file pickerではなくCMake project directoryである。Editorは管理build directory（`Library/UdonCL/Builds/<project-hash>`）へconfigureし、CMake File API等からartifact targetを列挙し、選択targetをbuild、Manifest検証、transactional importする。Preset UIは後回し。ONNX model、external weights、dynamic capacityはユーザーCMake project側で宣言する。

#### UdonBLAS acceptanceとONNX MLIR

- 最初の公開gateは独立CMake exampleから通常の`udonblasCreate`/`udonblasSgemm`/`udonblasDestroy`を呼び、`m=1,n=10,k=144`をUnity SDK IUdonVM上で実行すること。全144要素・10列に非ゼロ値を持たせ、CPU referenceとの10出力、最大誤差、9 tileのdispatch/readback/resume、reimport、console error 0、cleanupを検証する。UdonBLAS Unity専用`#ifdef`、SGEMM-only create、手書きUASMへ逃げない。
- UdonBLAS programは複数kernelを含むAOT programとしてManifest/Importerが扱う。`clCreateProgramWithSource`はlinked Device libraryの既知programを解決し、未知kernelを拒否する。`m/n/k`等はruntime値のまま維持し、libraryのprofile制限・tilingをONNX専用kernelで回避しない。最初は現在のC実装どおりtileごとにblocking readback/resumeして意味保存を優先し、GPU内累積は後続最適化とする。
- ONNXはexecution-plan JSONを生成・解釈しない。ONNX MLIRから直接`func.call`でUdonBLAS/UdonDNN/UdonATenを呼び、対応libraryがないoperatorだけcustom OpenCL Host/Device callへloweringする。所属component/interfaceはPyTorch/CUDA実装を確認して選ぶ。生成Host call graphを共通Host MLIR/continuation/Manifest経路へ通し、ONNX専用Material scheduler、texture arena binding、`static_inputs`を持たない。
- Dynamic ONNX dimensionは可能な限りruntime valueとしてlibrary APIとshape Host MLIRへ渡す。物理bufferだけ最大capacityを事前確保する。Initializer weightはmodel initializationで一度uploadし、immutable物理assetは共有可能だがlogical handle/queue/stateはinstanceごとに持つ。ONNXはmodel固有device lifecycleを持たず、通常のC Hostが`cl_context`/`cl_command_queue`を作成・所有してgenerated physical bindingへborrowさせる。生成側はmodel-derived I/O allocation/bind/sessionだけを所有し、Hostは標準`clEnqueueWriteBuffer`/`ReadBuffer`/`CopyBuffer`を使い、ONNX dispose後にqueue/contextを解放する。MNISTの公開Host lifecycleと同じ契約を全ONNX modelへ適用し、高水準ONNX Session U# APIをcompilerが固定しない。
- 実装順は、(1) `UDONCL_EXPORT`とschema v2、(2) `vrc` eventからMLIR Asyncへのcall graph変換、(3) upstream Async Runtime/Native LLVM coroutine、(4) Udon static frame/resume trampoline、(5) physical adapter/Manifest/低水準U# bindingのcompiler生成、(6) CMake/Importer統合、(7) MNIST C Host Unity gate、(8) 旧直接await/強制inline/Editor generator削除、(9) TinyLlama汎用経路、とする。
- 2026-11-03 checkpoint: installed CMakeにexecution-plan JSONを生成しない`udoncl_add_onnx_host_library`を追加し、MNIST ONNXのDense tailを`udonblasSgemm(m=1,n=10,k=144)`へ直接生成した。Unity 2022.3.22f1 / Worlds SDK 3.10.5の生成U# bindingで144入力、async完了、10出力を実行し、status 0、最大絶対誤差`3.814697e-6`、cleanup完了を確認済み。これはDense受入checkpointであり、MNIST全graphのmulti-kernel Host call graphとONNX MLIR lowering、Llama 8Bは未完了である。
- 同checkpointでinstalled `UdonDNN::UdonDNN` / `UdonATen::UdonATen` targetも追加し、生成MNIST Host C、UdonBLAS、UdonDNN、UdonATen、OpenCL AOT runtimeを重複定義なしに一つのHost LLVM moduleへlinkできることを外部CMake consumerで確認済み。次の阻害要因は複数Device kernelを一つのUnity library config/Manifestへ導出するcompiler側の対応である。
- `device_kernels`配列から同一Device LLVM module内の複数gather kernelを個別回復し、一つのUnity artifactへ複数Shader/HLSL/Manifest entryとして生成するcompiler対応まで実装済み。UdonBLAS SGEMM+SGEMVの2-kernel artifactで検証した。full MNISTの次の阻害要因はOpenCL AOT runtimeのkernel-id/argument ABIがSGEMM dispatch一つに固定されている点である。
- OpenCL AOT runtimeとtyped VRC loweringはSGEMV（kernel id 2、6個のi32と2個のf32引数）まで複数dispatch対応済みで、公開UdonBLAS/ONNX Dense artifactはSGEMM+SGEMVを一つのprogramとして生成する。full MNISTには同じ仕組みをUdonDNN convolution/pooling/pointwiseへ一般化する必要がある。
- 複数kernelで同一のMaterial property名をUdon heap slotとして共有し、SGEMM+SGEMV Dense HostをSDK上限内の479 data slotへ削減した。fresh Unity import、IUdonProgram取得、生成U# binding compileまで再確認済み。さらにexecution-plan JSONを使わず、ONNXから生成したHost Cが標準OpenCL APIで`mnist_conv`→`mnist_pool`→`mnist_dense`を呼ぶfull MNIST 3-kernel artifactを生成済みで、ローカルcompiler/runtime/CMake/device testは通過している。直接Loweringしたupload/readbackの2個の`vrc.await`について、同一SDK IUdonVMを3回起動してresume stateが`1 → 2 → 0`へ進むEditMode testも通過したため、前回Dense手動検証でstate 1を反復した現象はLowering本体ではなく検証ハーネス側の問題として切り分けた。さらに保存済みWorld sceneへ`VRCSceneDescriptor`、`PipelineManager`、spawn Transformを置いたClientSim PlayModeで生成Bindingの公開APIだけを使い、144入力のwrite、非同期callback 1回、10出力のreadが全status 0・timeoutなし・最大絶対誤差0で完了した。full MNISTのUnity数値実行は継続検証中であり、最終的なprofile固有generatorはONNX MLIR loweringへ置換する。
- ONNX MLIR移行の最初の実成果物として、MNIST Dense graphを`func`/`cf`/`arith`からraw heapの`udonhl.global`/`!udonhl.ptr<0>` ABIへlowerし、公開UdonBLAS ABIの`udonblasCreate`→`udonblasSgemm(m=1,n=10,k=144)`→`udonblasDestroy` call、model weight globals、3つの公開exportを生成する。`udonclc --host-library-llvm`で回復済みUdonBLAS/OpenCL runtime定義と型検査付きsymbol linkし、installed CMakeの`mnist-dense` profileはこのHost MLIRを標準入力として最終`host.uasm` artifactまで生成する。Host Cも数値oracle/移行互換出力として当面生成するが、artifact入力には使用しない。`vrc-opt --verify-each`、外部CMake consumer、artifact Manifest/assembly testを通過済み。次工程は同じlibrary-call loweringをMNIST全graphのUdonDNN callsへ拡張することである。
- Full MNISTもONNXの`Conv→Add→Relu→AveragePool→Flatten→Gemm→Add`を、公開`udondnnConvolutionForward`、`udondnnOpTensor`、`udondnnActivationForward`、`udondnnPoolingForward`、`udonblasSgemm`のHost MLIR call graphへlowerする候補成果物を生成する。9216-word convolution intermediate、descriptor setup、各statusのCFG分岐、error/success共通cleanupを含み、`vrc-opt --verify-each`とinstalled CMake consumerを通過済み。現行public UdonDNN実装は各operatorの512-word staging制限がありこのshapeをまだ実行できないため、専用arena APIへfallbackせず、大容量を扱う公開UdonDNN backendが数値検証されるまではfull MNIST artifactの標準入力を既存互換Host Cから切り替えない。
- 公開`udondnnPoolingForward`は512 wordsを超える入力に対して16,384-word固定Device bufferを使う通常のOpenCL C backendへ分岐する。MNISTの`1x16x24x24 → 1x16x3x3`、8x8 AveragePoolを含む汎用window/pad/stride/mode/alpha/beta ABIを15 kernel引数へpackし、CPU reference数値一致とUdonCL Device bundle lowering（8192x2）を検証済み。これはprivate ONNX APIやexecution-plan fallbackではない。Full MNIST切替には、残る大容量convolution、op-tensor、activationの公開backendとmulti-kernel Host continuationの数値検証が必要である。
- 公開`udondnnConvolutionForward`も512 wordsを超えるtensor/filter/outputを16,384-word固定Device bufferへ転送し、通常のOpenCL C backendを呼ぶ。MNISTの`1x1x28x28`入力、`16x1x5x5` filter、`1x16x24x24`出力について、公開APIのalpha/beta、convolution mode、stride/pad/dilationを16 kernel引数へpackし、9,216出力のCPU reference一致とUdonCL Device bundle lowering（8192x2）を検証済み。残るFull MNISTのoperator容量阻害は大容量op-tensorとactivationである。
- 公開`udondnnOpTensor`と`udondnnActivationForward`にも16,384-word backendを追加した。OpTensorはNCHW各座標をsource dimensionで剰余するため、MNISTの`1x16x1x1` channel biasを`1x16x24x24`へ正しくbroadcastし、ADD/MUL/MIN/MAX、NaN propagation、alpha/betaを保持する。Activationは全公開modeとNaN propagation、coefficient、alpha/betaを保持する。9,216-word Add→ReLUのCPU reference一致と、2 kernelのUdonCL Device bundle lowering（8192x2）を検証済み。これでFull MNISTのoperator容量阻害は解消し、残件は公開call graphのmulti-kernel Host continuationとUnity数値gateである。

### 2.6 汎用ONNX MLIR runtimeとTinyLlama 1.1B

TinyLlamaの現行設計、実装順序、受入条件の正本は
[`TINYLLAMA_AOT_CONVERSATION_PLAN.md`](TINYLLAMA_AOT_CONVERSATION_PLAN.md)とする。
この節の古いsequence=4固定、tokenizer対象外、INT64 Gather阻害という記述は廃止した。

- ONNX Dialect import後のgraph検証、canonicalize、shape処理、operator選択、
  bufferize、Host/library call生成はC++ MLIR passで行う。Python graph parser、
  execution-plan JSON、model profileへfallbackしない。
- 実modelは固定revisionの
  `onnx-community/TinyLlama-1.1B-Chat-v1.0-ONNX/onnx/model_q4.onnx`
  （909750422 bytes、SHA-256
  `399360b3b9174b505eedfc71a305f48b98207c51fff4c4643c722b7e37f137eb`）とする。
- 次の阻害点はGatherではない。最優先は、MNISTとTinyLlamaで共通に使う
  immutable static Device tensor AOT、Device-resident `MatMulNBits`、workspace/KV
  bufferizationである。
- `gather-f32`の現行証明view上限は32,768 word（128 KiB）だが、immutable
  texture自体は8192x8192 RGBA32まで検証済みである。論理/物理static tensorを
  分割せず、resource byte length・texture width・base/lengthを持つtyped tile viewを
  導入する。`MatMulNBits`はweight/scale/zero viewとDevice partial/outputを
  複数dispatchで処理し、Hostでのweight repackとpartial readbackは禁止する。
- 実TinyLlamaはONNX Dialect→portable Host MLIR→finalized Host MLIR→compact Unity
  bundle生成まで到達し、Host data slotは442/512、immutable tensorは1,033個・約909 MB
  になった。ただし完全Device graph未対応のためHost fallbackとなり、raw heapへ約939 MBを
  重複しており推論成功とは扱わない。compilerはoperator countと未対応providerを記録し、
  `require_device_graph`指定時はこのfallbackを早期errorにする。
- ONNX initializerは原則1 initializer＝1 lossless Device textureとしてartifactへ
  生成し、Host raw heapやsession初期化uploadへ複製しない。activation、intermediate、
  KV、logitsもoperator間でDevice residentに保つ。
- Device intermediateはcompile-time livenessと必要byte容量でallocationを再利用し、
  同一dispatchの入力slotは出力に再利用しない。large convolution/poolingのlaunchは
  8192x2、large activation/op-tensorは8192x4としてDevice configとHost loweringを一致させる。
- gather-profileはread-only inputごとに独立した証明済みword容量を持ち、memref、
  Texture幅、HLSL座標、manifestの`required_bytes`へ保持する。Q4向けtyped viewの
  read-only inputごとのcapacityに加え、i32 scalarのinclusive min/maxを
  Device index証明へ渡し、`base + gid`がcapacity内であることをcompile時に
  検証する。証明済みminimum/maximumはmanifestのscalar argumentにも保持し、
  生成dispatch側のruntime検証契約とする。dynamic length、manifest rangeの
  runtime enforcementに先立ち、`const uint*` inputをRGBA8のlittle-endian
  byte列から`uint32`へ復元するraw typed viewを実装済みで、Q4 bit extractionを
  floatへの誤変換なしでshader内に保持できる。さらに実modelの3-input
  unsigned Q4形（bits=4、block=32、implicit zero=8、FP32、M<=4、K<=5632、
  N<=32000）向けに、immutable packed/scalesを直接読むkernel 235と、全N列を
  1 dispatchで処理するK=128 compile-time tile、131,072-word ping-pong surface、
  動的resource byte検証、最終public output copyを
  実装済みである。小型static-initializer ONNX graphでHost生成・allocation・dispatch・
  disposalを検証し、実sequence-4 graphの155個すべて（N=32000のlm_headを含む）が
  Device Q4契約へ入ることも確認した。full graphの他operator、dynamic sequence、zero-point付き、
  bits 2-8/FP16/BF16/g_idx/biasのgeneric Device対応、sub-word/unaligned viewは未完了である。
- 同じDevice graph finalizationで、UdonATenの2-pass arena RMSNormをstatic epsilon、
  immutable weight、共有operator scratchへ接続し、実graphの45 nodeすべてをHost readbackなしで
  対応した。さらにUdonDNN arena Softmaxを`outer x axis x inner`へ汎用的に写像し、
  16,384 word以下・reduction 512要素以下のstatic FP32 tensorを対応した。standalone Device
  数値suiteと実graphの22 nodeを検証済みである。UdonBLASにはさらに16,384 word以下・
  K<=256のstatic FP32 strided-batched SGEMM Device backendを追加し、input stride 0のbroadcast、
  contiguous output、実attention形状をNative数値testで検証した。実graphのMatMul 45 nodeも
  すべてこの契約へ入る。さらにstatic contiguous FP32、rank 1..5、16,384要素以下の
  Transposeを汎用Device reindex kernelへlowerし、実graphの111 nodeすべてを対応した。
  static parameter、正規化後startが非負かつstepが正のFP32 Sliceも同じ汎用Device
  reindexへlowerし、実graphの111 nodeすべてを対応した。さらにrank 0..5、16,384要素以下の
  static FP32 Expandをright-aligned broadcastの汎用Device reindexへlowerし、実graphの46 nodeを
  すべて対応した。2入力のstatic contiguous FP32 Concatも単一の汎用Device dispatchへlowerし、
  render targetを読み戻さず実graphの45 nodeをすべて対応した。immutable FP32 tableの
  axis-0 GatherはINT32/INT64 indexをraw wordとして読み、負indexも含めて安全なtexture addressへ
  正規化する汎用Device kernelへlowerする。実graphのembedding Gather 1 nodeも対応済みで、
  INT64→FP32 Castもlittle-endian raw wordから64-bit shader arithmeticなしで変換する
  汎用Device kernelへlowerし、実graphの2 nodeを対応した。FP32 EqualとFP32 Whereも
  rank 0..4のright-aligned broadcastを行う汎用Device kernelへlowerする。論理的なONNX
  BOOL dtypeは維持しつつ、Device上ではword-addressableなFP32 0/1 maskとして物理化する。
  実graphのEqual 1 nodeとWhere 1 nodeを対応した。最後にreduction=none、rank 1..4、最大256
  update tupleのFP32 ScatterNDを、各fragmentが1出力を所有してupdateをONNX順に走査する
  汎用Device kernelへlowerした。atomicやrender-target readを使わず、重複indexは後勝ちになる。
  これによりsequence=4 / past=0の実graphはunsupported Device operator 0になった。full artifact
  生成はDevice graph gateと35 mutable allocation siteのpool gateまで通過し、現在の阻害はtoy C Hostの
  3-input/3-output bindingと実modelの47-input/45-output bindingの不一致、およびalias outputの
  full-surface CopyBuffer契約である。Q4 logitsの公開bufferは131,072-word物理surface上の論理viewとして
  扱う方針に更新済みである。
- TinyLlama tokenizerはONNXではない。独立したC `UdonTokenizers` component、
  `udon_string_t`/caller-owned `udon_int32_array_t` ABI、allocation-freeな
  UTF-16/byte-fallback BPE encode core、専用Host-object lowering、CMake package、
  compiler生成`string`/`int[]` U# bindingまで実装済み。次はbuild-time TinyLlama
  table生成、decode、special token/chat templateと公式tokenizer conformanceを追加し、
  C Hostの会話APIから利用する。
- 最初のdeploymentはPC/D3D11、batch 1、context capacity 128、最大32生成tokenとし、
  2-turn対話でKV再利用を検証する。全decode stepの32,000 logitsと22層K/VをORTと
  比較する。
- 実modelはcontent-addressed cacheを使うが、最終precheckは毎回oracle、Native、
  Device、generated-U# Unity testを実行する。Unityはlocal precheck、非Unity gateは
  GitHub Actionsでも実行する。
- 新経路合格後に旧execution-plan、ONNX専用Material/arena scheduler、
  `static_inputs`、toy TinyLlama consumerを削除する。

### 2.7 Host storage allocator（Grilling 2026-11-05）

詳細な実装仕様・コスト根拠・段階・テスト・受入条件の正本は
[HOST_STORAGE_ALLOCATOR_PLAN.md](HOST_STORAGE_ALLOCATOR_PLAN.md) とする。

- Whole-program promotion後に、固定公開slot、typed scalar register、物理型別typed spill array、
  word-packed raw memoryの4層へ分類する。初期backendは非再帰・1 active rootに限定する。
- Suspendをまたぐ値はpersistent typed spillへ保存し、register、spill index cache、raw fallback、
  externを含む最終address数を512以下にする。超過時は内訳付きcompile errorにする。
- generated U# APIだけでMNISTのinitialize、784要素input write、callback-first
  run、10 logits read、shutdownを行うUnity Test Runner数値試験は合格し、10秒watchdogも解消した。
  次のgateは`DEVICE_RESOURCE_ALIAS_PLAN.md`に従い、同じ試験をHost stagingのない
  CUDA-shaped asynchronous Device library APIへ移行した後にも通すことである。

## 3. ABIとExecution Contract

### 3.1 Host C ABI

- Target `vrc-udon-unknown`（仮名）。整数のサイズ、bitfield layout、struct packing、long double、`va_list`, `setjmp`などは**Clang TargetInfo/DataLayoutでテストベースに確定**すること。`int/long/pointer=32`, little-endianは決定済みであり、host tripleを偽の既存ターゲットに紛れ込ませない。
- Heapのbyte addressing、effective type、strict aliasing、volatile、`memcpy`, unaligned access、signed/unsigned conversionをC仕様に沿って実装する。SSA promotionとtyped register/spillは物理最適化でありABIの公開型ではない。observable byte memoryが必要なobjectはlittle-endian word-packed raw heapへ戻す。
- メモリ領域の割当表は`base: u32`, `size: u32`, `kind: raw|typed`, `generation`, `backing`, `permissions`を持つ。`ptrtoint`/`inttoptr`後のprovenanceが曖昧ならallocation lookup + byte-oriented semanticsへ戻し、aligned word fast pathとunaligned byte head/tailを区別する。uintptr overflow/invalid memory accessはDebug診断と必要なRelease安全チェック。
- `cl_mem` / `cl_kernel` / `cl_program` / `cl_event` opaque handlesは各tableでタイプ検証（ゼロやstale IDは無効）。Host flat pointerとGPU virtual pointerとhandleを混同しない。Entry/return framesは同一library instanceに属する。

### 3.2 U#↔C Export ABI

- Manifestに`export_id`, `public_name`, `udon_event_name`, `parameter_slots`, `result_slots`, `status_slot`, `async`, `types`, `required_services`を記載。Udon assembly `.export`（event）/`.data`変数名とManifestが一対一に対応する。
- **同期**: U# Wrapperがbusy/initializedを検査、input slotsへ値を設定、`SendCustomEvent(event)`実行、result/statusを読む。メソッド実行がイベントを戻るまでに終わることを同期APIの前提とし、実機で検証。
- **非同期**: Startメソッドは`StartStatus`（accepted/busy/argument_error等）を返し、GPU callbackでContinuation復元後、status/result slotsを確定→busy解除→指定U# callback event送信。Callbackが別の呼出を開始してもスロット内容破壊なし（完了データのsnapshotまたは再呼び出し前の取得契約を明示）。
- C function `void`/scalar/arrays/strings/struct marshallingについてTypeSchemaを定める。U# Dataで表現不能なC型（function pointer等）はExport禁止し、C内部では可能ならサポート。
- `clFinish`, blocking read/writeなどC API上の“blocking”は**論理的にC呼出の完了まで待つ**ことであり、Udon VMイベントを同期ブロックする意味ではない。Host ContinuationにLoweringする。

### 3.3 非同期とGPU順序

- In-order Queueのcommand sequence id、pending requests、resource lifetimes、failure states、generation idをC Runtime側で保持。
- Compute Blit列→最後に1x1 fence RenderTextureへBlit→`VRCAsyncGPUReadback.Request(fence, ...)`→`OnAsyncGpuReadbackComplete`でC Continuation再開するのは**検証待ちの方式**。GPU APIの呼出順が正しく、fence Readbackが先行パス全ての完了を意味するかをGPU実機でテスト。検証できないなら適切なReadback / completionを使う別方式に切替。
- `buffer_readback`で要求されたデータのReadbackは別に管理。Readback requestとContinuation idの対応付けを確認できない場合、要求を直列化する。
- `VRCGraphics.Blit` sourceとdestinationの自己参照は使用禁止。Ping-pong、Buffer version追跡、前版の不要回収を実装する。
- 同一C libraryの実行中に新規C APIを呼ぶBusy制約と、in-order Queue内の複数pending GPU commandは別問題であり両立する。

## 4. コンパイラのFrontend・LLVM IR復元

### 4.1 High-level MLIR Recovery（最重要）

Grilling 12の決定は**方式C：LLVM IRからHigh-level MLIRへの包括的な復元**。簡単な部分だけをraiseする「選択的復元で十分」とみなさない。実際にはLLVM IRは元のループ/配列/Work-group意図の一部を失い得るため、情報を**再構成できる条件を証明する**。未証明の推測や意味の変更は禁止。

- `clang -O0`または復元向け最小最適化、Debug/Kernel Metadata/OpenCL address space/kernel arg情報を保存する**sidecar**をFrontendから作る。`optnone`、alloca形、標準LLVM importの対応制約との関係は比較検証する。
- LLVM IR importerは限定的/実験的な取り込み経路を含むため、現在のLLVM/MLIR版で必要なIR/Metadata/Calling ConventionがImportできるかを先に確認する。失われる情報をSidecarに保管し、必要なら合法なpre-normalizationまたはImporter extensionを実装。
- 独立ライブラリ **`VRCLLVMToHighLevelMLIR`**（Host/Device共通）。Pass: CFG/loop recovery、scalar evolution/induction、memref access/subscript recovery、alias/provenance、affine access、GPU launch/work-item builtin recovery、addrspace/local memory recovery、barrier/dependence proof、unknown structure diagnostics。LLVM Dialectのまま残すOpを安全にLoweringできない場合はエラー。`-verify-each`/Pass単体テストを提供。
- `gpu/scf/memref/affine`の全適用はケースに依存。`affine`にできない非affine loopも`scf`として復元し、解析できる範囲で保存する。失敗理由と元のLLVM IR locationを診断。
- OpenCL SPIR-style metadata・builtinsの名前解決はDataLayout/Clang version差を考慮してGolden Test。Kernel entry/argumentsはManifestに保存する。

### 4.2 Host pipeline

`Clang → LLVM IR (host + C Runtime + libraries static link) → Import → Common High-level Recovery → export-aware internalization → whole-program SROA/mem2reg/inlining/constant propagation/descriptor specialization/DCE → ABI/alias analysis → vrc intrinsic substitution → may-suspend call graph to MLIR Async → SCF-to-CF/canonicalize → upstream async-func-to-async-runtime → Native LLVM coroutine or Udon static-frame/trampoline legalization → udonhl → 4-layer storage planning/register allocation → Udon extern legalization → udonasm → verifier → .uasm + export ABI manifest + binding.cs + allocator report`。

順序上、promotionとwhole-program specializationをstorage planningより先に完了し、非同期Call Graphとlive-value analysisを非再帰interprocedural register/spill allocationへ渡す。**GPU callbackで従来のUdon event stackは残らない**ことを前提にし、再帰SCCは初期backendで拒否する。

### 4.3 Device pipeline

`Clang OpenCL C 1.2 → LLVM IR + sidecar → Import → High-level Recovery → OpenCL memory model + NDRange legality → Dependence/alias analysis → barrier uniformity proof → barrier elimination / gather substitution → cost-aware phase split / local scratch texture versions → output ownership & scatter rejection → logical byte memory → texture formats/bindings legalization → vrcgpu passes → HLSL + ShaderLab + Manifest`。

- 高レベルOpenCL GPU意味論を`vrcgpu`まで引きずらない。`vrcgpu`は**Fragment-safe**なpass表現であり、barrier, local/shared GPU memory, work-item write scatterを含めない。
- `vrcgpu`のHLSL Emitterは直接実装。**SPIR-V/EmitCを必須経路にしない**（拡張余地は残す）。ShaderLabパス、fragment signature、Texture formats、binding names、Uniform、bounds、output bytes、versionsを生成。
- 動的loopはHLSLで安全に生成できれば許可。GPU長時間実行の警告。work-group barrierを伴うcompile-time local size specialization以外のKernelでは実行時サイズを維持できるようにする。
- Reductionはatomic scatterに変換しない。各passの各fragmentが一意な出力を持つtree/pairwise形式に変換し、group間まで繰り返す。中間passは別Textureへ書く。

## 5. Dialect実装規約・共通属性

**ここから§6〜§9が4つの独自Dialectの全Operation一覧（初期v0.1のclosed catalog）。** 実装開始後に追加・分割する場合、**この一覧、TableGen、Verifier、docs、testsを同一PRで同期変更**する。MLIR標準Opを別名で重複実装しない。

- 共通属性: `vrc.abi_version`, `vrc.target`, `vrc.source_loc`, `vrc.feature_level`, `vrc.sdk_extern_db_hash`, `vrc.export_id`, `vrc.provenance_id`, `vrc.memory_kind`, `vrc.async_effect`, `vrc.debug_id`。所有Dialectの属性として定義するかModuleの字句namespaceとして扱うかは衝突しない実装を選び、一度規約化する。
- Common types: `i1`, `i8`, `i16`, `i32`, `f32`, `f64`, `index`, `vector`, `func`等標準MLIR type。OpenCL側で論理`global/local/private/constant` address spacesの情報は復元中に維持する。
- 各OpはTableGen ODSで**operand/result、attributes/properties、memory effects、traits、region/verifier、assemblyFormat**を定義。`mlir-tblgen`のMarkdown documentationも生成する。
- below `!` types/`#` attributesは**設計する独自構文（説明用略記）**。完全なパーサ実装済みを意味しない。表のLoweringは特定のBackendへの責務を示す。Verifierでは`IsolatedFromAbove`、symbol table、pass program order、memory accesses、SSA legality、resource bindingなどを検査。

## 6. `udonhl` Dialect — 全Operation一覧（23 Op）

**位置付け**: Cとして意味のあるメモリ/ポインタ/関数/非同期フレームとUdon最適化の接点。`func.func`, `func.return`, `arith.*`, `math.*`, `cf.*`, `scf.*`は**再利用**し、独自定義しない。`!udonhl.ptr<addrspace>`は意味論上u32のbyte addressであり、provenanceはOp/Valueの属性・分析サイドテーブルに置く。`!udonhl.frame`はContinuation実装のstate object。`#udonhl.memkind<raw|typed|unknown>`、`#udonhl.suspend_reason<gpu|budget|service>`、`#udonhl.call_policy<normal|recursive|may_suspend>`を定義する。

| # | 完全名 | 主要Operands → Results / attrs | 意味・Verifier / Lowering |
|---:|---|---|---|
| 1 | `udonhl.global` | symbol, type, optional initializer, linkage | Instance-local C global/BSS定義。定数・storage size・duplicate symbolを検証。`.data`/init codeへ。 |
| 2 | `udonhl.address_of` | global symbol → `!udonhl.ptr` | Globalの論理byte address。symbol existenceとaddrspaceを検証。 |
| 3 | `udonhl.alloca` | byte_size, alignment → `!udonhl.ptr` | C automatic storage。frame/stack lifetime内のみ。 |
| 4 | `udonhl.heap_alloc` | size, alignment → `!udonhl.ptr`, success | Raw/typed memory allocator backend intrinsic。C `malloc`実装と競合しないよう契約を明確化。 |
| 5 | `udonhl.heap_free` | `!udonhl.ptr` → void | 生成した論理メモリallocの解放。ownership/release整合性。 |
| 6 | `udonhl.ptr_add` | ptr, signed/unsigned byte offset → ptr | C byte pointer arithmetic。overflow・provenance追跡。 |
| 7 | `udonhl.ptrtoint` | ptr → i32 | C `uintptr_t`への変換。provenance lossを保守的に処理。 |
| 8 | `udonhl.inttoptr` | i32 → ptr | integer→pointer。unknown allocationならmap lookup/raw。 |
| 9 | `udonhl.load` | ptr, access_type, align/volatile → scalar/vector | C load。アラインメント・endianness・backing fallback・byte semanticsを保持。 |
| 10 | `udonhl.store` | value, ptr, align/volatile → void | C store。同allocation内の型pun処理でrawへ。 |
| 11 | `udonhl.memcpy` | dst, src, size → void | Non-overlap前提。必要ならbyte copyの組み立て。 |
| 12 | `udonhl.memmove` | dst, src, size → void | Overlap可、方向またはtemporaryを保証。 |
| 13 | `udonhl.memset` | dst, i8 byte, size → void | byte単位のfill。 |
| 14 | `udonhl.lifetime_start` | ptr, size → void | Optimization hint/stack slot再利用用。C動作を変更しない。 |
| 15 | `udonhl.lifetime_end` | ptr, size → void | 上に同じ。終了後アクセスを正当化しない。 |
| 16 | `udonhl.array_view` | Udon typed array reference, element metadata → ptr | **条件付き共有/zero-copy**: Udon arrayをC byte-backed allocationとして一貫view。型/要素/寿命/aliasを検証。fallback copy。 |
| 17 | `udonhl.call` | callee symbol, operands → results; call_policy | C function call。SCC解析、再帰live-values spill、may_suspend伝播。通常callはstatic slots/inlining可。 |
| 18 | `udonhl.frame_create` | frame_layout, execution_id → `!udonhl.frame` | Continuation/call frame作成、slot layout version検証。 |
| 19 | `udonhl.frame_store` | frame, slot, value → void | Suspend前のlive value/return address/locals保存。型とslot indexを照合。 |
| 20 | `udonhl.frame_load` | frame, slot → value | Resume時復元。未初期化slot禁止。 |
| 21 | `udonhl.resume_dispatch` | frame/state → successor blocks | **terminator**。state idからResume先を選択、`cf.switch`やUdon jumpsにlower。 |
| 22 | `udonhl.suspend` | frame, reason, resume_state, event/budget token? → none | **terminator**。Udon eventを終了し次callback/frameで再開。必ずlive state永続化済み、destinationは対応resume dispatchに登録。 |
| 23 | `udonhl.ref_cmp` | managed lhs, rhs; `eq`/`ne` → i1 | Address space 200/201のUdon object参照を型を保ったまま比較する。Udonでは`SystemObject`参照比較へlowerし、`ptrtoint`や整数slotへのCOPYを禁止する。順序比較は明示error。 |

**補足**: `udonhl`には独自`add/mul/if/for/return/branch/function definition`を作らない。`udonhl.trap`も初期Opではなく、必要な診断/停止はRuntime service + `cf` loweringで表現する。U# Export entry stubは`func.func`の`vrc.export_id`, `udon_event`等の属性で示す。`array_view`の一貫したC byte semanticsを証明できない場合はzero-copyしない。

### `udonhl`の重要不変条件

1. `udonhl.load/store`は完全なC byte memory modelに従う。Scalar promotion、typed register、typed spill arrayは最後のloweringの物理最適化であり、strict aliasing未証明を理由に結果を変えない。
2. 全`udonhl.suspend`は対応するresume stateを持ち、live scalarをpersistent typed spillへ保存済みである。永続化不可能なUdon object handleを持ち越すケースはVerifierで拒否し、再帰stackは初期backendの対象外とする。
3. 非同期関数のcallee/callerには`may_suspend`が伝播、同期Export wrapperが待機する関数を誤って呼ばない。
4. 32-bit pointerでC ABIを表現するが、`i32`とptrを型として分ける。任意ポインタを最適化専用typed addressへ誤変換しない。

## 7. `udonasm` Dialect — 全Operation一覧（9 Op = 実VM opcode 9種類）

**唯一の意味**: 実際のUdon VM opcodeと**厳密に1:1**。擬似CALL, RETURN, LOAD, STORE, ALU, AWAITなどの命令を追加しない。Udon VM公式掲載の命令番号は`0,1,2,4,5,6,7,8,9`で、`3`は使用しない。`.data_start/.data_end`, `.code_start/.code_end`, `.export`, `.sync`, Heap declaration, code labelは**assembler directive / module metadata / symbol**であり**Operationではない**。

| # | 完全名 | VM opcode | オペランド | 意味・Verifier |
|---:|---|---:|---|---|
| 1 | `udonasm.nop` | 0 | none | No-op。ラベルaliasing回避等に使用可。 |
| 2 | `udonasm.push` | 1 | heap slot symbol/index | **値でなくHeapアドレス**（u32）をVM stackへpush。 |
| 3 | `udonasm.pop` | 2 | none | VM stack topを破棄。Underflow禁止。 |
| 4 | `udonasm.jump_if_false` | 4 | code label/address | Stack topからHeap address→boolを読みfalseならjump。スタック型・label解決を検証。 |
| 5 | `udonasm.jump` | 5 | code label/address | 無条件jump。終了sentinel `0xFFFFFFFC`も許す。 |
| 6 | `udonasm.extern` | 6 | Extern signature heap slot/string symbol | Externを実行。積まれたHeap slot順、this/args/out/result slotをSDK DBで確認。 |
| 7 | `udonasm.annotation` | 7 | u32 parameter | 注釈No-op。結果への影響なし。 |
| 8 | `udonasm.jump_indirect` | 8 | heap slot symbol/index | 指定slotに格納された`SystemUInt32`をコードアドレスとしてjump。 |
| 9 | `udonasm.copy` | 9 | none | `PUSH src_slot; PUSH dst_slot; COPY`でsource value→destination。順序をVerifier/Golden Testで保証。 |

**補足（Opcode 1:1に必要な設計）**:
- Udon VMにnative function call/return/localsはない。High-levelの非再帰callは`JUMP` + 戻り先`SystemUInt32` + `JUMP_INDIRECT`へlowerし、call graph全体の静的interferenceでregister/spillを割り当てる。再帰SCCは初期backendで拒否し、将来のdynamic frame対応まで通常callへ混ぜない。
- 値計算・配列操作はSDK Externを使い、`PUSH` sequence→`EXTERN`へ。`udonasm.extern`は**単一のVM命令**であり、引数PUSHを隠すHigh-level pseudo-opにはしない。
- `.data` declarations: name / Udon type（例`SystemInt32Array`）/ initial value / visibility/sync mode。固定公開slot、型付きscalar register bank、物理型別spill array、word-packed raw heap、Global Init、Export ABI stubをstorage plannerが生成する。labelsとcode addressesはAssembler/relocation passが付ける。`udonasm`自体はSSA型の値レジスタを導入しない。
- Verifier: 全jumpのアドレス整列/範囲、data/Extern slots、stack effects、Extern signature、Export名衝突、同位置label aliasing、Return sentinel、安全なcallback resume入口。

## 8. `vrc` Dialect — 全Operation一覧（13 Op）

**意味**: C Runtimeで表現不能なVRChatサービス境界、GPU Command I/O、非同期イベントをHost Compilerへ露出する。OpenCLのハンドル／コマンドキューの高位ロジックはC Runtimeに属し、`vrc`は`__vrc_*` Intrinsicに対応。`!vrc.buffer`はGPU allocation handleのtyped IR wrapper（ABI境界はu32）、`!vrc.event`はpending completion token、`!vrc.service`はU# service binding、`!vrc.dispatch_desc`はKernel Manifestに従うdescriptor。必須Op以外のKernel管理はCで実装。

| # | 完全名 | 主要Operands → Results | 意味 / Verifier / Lowering |
|---:|---|---|---|
| 1 | `vrc.buffer_alloc` | size, format/usage → `!vrc.buffer`, status | 物理Texture backingを用意する。CPU側`cl_mem`とは別ID。サイズ/format/上限チェック。 |
| 2 | `vrc.buffer_release` | buffer → status | GPU参照が残るときはdefer。resource generation check。 |
| 3 | `vrc.buffer_upload` | buffer, offset, size, host_ptr/array_ref → `!vrc.event` | Host→GPU。Stage Texture候補、Q18で計測。format/offset一致。 |
| 4 | `vrc.buffer_readback` | buffer, offset, size, host_dst → `!vrc.event` | GPU→Host Async Readback + byte/typed unpack。data lifetime/boundsを検証。 |
| 5 | `vrc.buffer_copy` | src, dst, src_offset, dst_offset, size → `!vrc.event` | Device-internal buffer copy。必要ならpixel pass、overlap時は別temporary。 |
| 6 | `vrc.dispatch` | kernel_id, desc, ordered resource handles → `!vrc.event` | Manifest-defined multipassをQueueへ投入。arg types/binding/pass/formatを検証。 |
| 7 | `vrc.fence` | queue/sequence → `!vrc.event` | QueueのGPU完了待ちに用いるfence request。Blit→Readback方式は実機検証待ち。 |
| 8 | `vrc.await` | event → completion status | **may_suspend**。`udonhl.suspend`/Continuationへ変換し、Udon call stackでは待たない。 |
| 9 | `vrc.event_status` | event → status | Completed/error/cancelled/pending判定。event generation/ownership検証。 |
| 10 | `vrc.service_call` | service_id, opcode, args → status/results | U# service (stdio/VFS/time/log等)との同期Boundary。非同期serviceは初期非対応。 |
| 11 | `vrc.notify_complete` | export_id, status/result slots → void | U#完了Event通知。busy cleared、snapshot/stable result条件を検証。 |
| 12 | `vrc.notify_failure` | export_id, error code → void | GPU失敗/Runtime errorの完了通知。成功と同じ完了経路・status別。 |
| 13 | `vrc.budget_check` | execution budget/counters → i1 | CPU safepoint実行可否をチェック。false→`udonhl.suspend` (budget) path。 |

**Important semantics**:
- 上記各OperationはC OpenCL APIの代用品ではない。`clCreateBuffer`/`clEnqueueNDRangeKernel`等は**C Runtime function**であり、内部`__vrc_*`だけをvrc化する。
- `vrc.buffer_upload`等が返すeventは論理的command completion token。GPUからの結果回収要求と`vrc.fence`の完了意味論が異なることに注意する。
- `vrc.await`を含む関数の`may_suspend`は呼び出し元へ伝播。`vrc.service_call`は初期同期のみ。`notify_failure`から二重callbackしない。
- GPU command中のTextureは`clReleaseMemObject`後も物理リソースを保持できるようRefcount + in-flight tracking。

## 9. `vrcgpu` Dialect — 全Operation一覧（13 Op）

**意味**: OpenCLの計算を合法化し終えた**Fragment Shaderの最終高レベルIR**。1 output pass、決定済みリソースBinding、byte/typed texture access、pass間SSA dataflowを保持。`!vrcgpu.texture<format, access>`、`!vrcgpu.uniform<T>`、`!vrcgpu.target<format>`、`!vrcgpu.version<buffer-id>`を定義する。`#vrcgpu.format<rgba8_unorm|rfloat|...>`、`#vrcgpu.access<read|write>`, `#vrcgpu.layout<...>`, `#vrcgpu.pass_cost<...>`。`scf`, `arith`, `math`および合法な`vector` Opは再利用。

| # | 完全名 | 主要Operands → Results / attrs | 意味 / Verifier / HLSL mapping |
|---:|---|---|---|
| 1 | `vrcgpu.kernel` | symbol, entry metadata, Region containing resources/passes | AOT Kernel集合のsymbol。合法なpass/metadataとManifest対応を保証。 |
| 2 | `vrcgpu.resource` | symbol, kind=texture/uniform/target, format, binding, access | Kernelの論理Buffer/Uniform/Render Target宣言。Binding衝突、format整合性、target用途を検証。 |
| 3 | `vrcgpu.pass` | read-version SSA operands + binding operands; Region; → written version token(s) | **1 Region = 1 Fragment pass = 1 output Texture**。依存SSAはversion化し、循環・同Texture source/destは禁止。Region argsはbound resources/uniforms。 |
| 4 | `vrcgpu.fragment_coord` | none → `(u32 x, u32 y)` | 出力pixelの座標。HLSL `SV_POSITION`から計算。 |
| 5 | `vrcgpu.logical_index` | coord, texture_width, logical_origin/stride → u32 | 2D→logical linear index、work-item対応範囲へのmap。integer overflow検証。 |
| 6 | `vrcgpu.launch_param` | `param_kind` → scalar | NDRange global size/offset/dimension/local-specialization等のUniform情報を読む。 |
| 7 | `vrcgpu.uniform_load` | uniform symbol/ref → scalar/vector | 固定BindingからUniform値を取得。type/layout検証。 |
| 8 | `vrcgpu.texture_extent` | texture → `(width,height)` | Logical/physical Texture寸法。Manifestと一致。 |
| 9 | `vrcgpu.texture_load` | texture, 2D coord, mip=0 → texel vector/scalar | HLSL `.Load()`相当。bound/format/point-access semantics。Samplerなし。 |
| 10 | `vrcgpu.raw_unpack` | packed texel(s), byte_offset, target type → value | RGBA8内のbyte addressing、little-endian解釈、符号/bitcast。誤ったUNorm変換やsRGB禁止。 |
| 11 | `vrcgpu.raw_pack` | value/old texel, byte_offset, layout → packed texel | Byte store/output用encode。部分byte更新では隣接byteを保全。 |
| 12 | `vrcgpu.in_bounds` | logical index, logical element/byte extent → i1 | 余剰pixelへの書き込み禁止、部分更新のpreserve/guardに用いる。 |
| 13 | `vrcgpu.output` | texel/scalar value → none | **Pass Region terminator**。唯一のRender Targetへの出力。ShaderLab `SV_Target`へ。 |

**重要なIR制約**:
1. `vrcgpu`には`gpu.barrier`, OpenCL `__local`, `get_global_id`, atomic scatter, arbitrary dynamic allocation, device pointer dereferenceを残さない。必要な情報は具体的なfragment coordinate / read ownership / texture bindingに変換済み。
2. Passは**単一RenderTarget**。複数論理出力は複数Passとバージョン依存へ。MRTを前提にしない。
3. `texture_load`は「出力座標に紐づく読取専用Texture」を読む。書込先Textureを同Passで読まない。Ping-pongはRuntime physical bindingで実装。
4. `vrcgpu.pass`はRegionの入出力とwrite-versionにより依存関係を表現する。`vrcgpu.pass_dependency`や`texture_version`という追加Opは作らない（SSA tokenで十分）。
5. 新しい arithmetic Opを`vrcgpu`に追加せず標準DialectをEmitterがサポートする。Shaderに非合法な標準Opが残ればVerifier error。
6. `raw_pack/unpack`は**Texture pixelレイアウトの変換**であり、C/OpenCLのhost/device論理メモリ型とは別概念。対応Texture formatsは実機実証したものだけ公開。

## 10. Operationカタログの境界・Loweringと「全Op」の解釈

- 独自Dialect v0.1の完全一覧: `udonhl` **23**, `udonasm` **9**, `vrc` **13**, `vrcgpu` **13**、合計**58 Op**。この文書で明記しない独自Opは**初期仕様に存在しない**。新規追加は設計改訂＋対応するlowering/Verifier/Testを必須とする。
- 標準MLIR Op全体を57に数えない（`func`, `arith`, `math`, `cf`, `scf`, `gpu`, `memref`, `affine`, `llvm`, 必要なら`vector`等）。自前再定義はしない。Importer/Raising中の標準Dialect OpについてもLegalization/Passごとのallowlistを設ける。
- `udonhl.call` + `func.func` → Udon static call slots / recursive frame code → `udonasm.push`, `udonasm.copy`, `udonasm.jump`, `udonasm.jump_indirect`、data/metaはExport directiveへ。
- promotion後に残る`udonhl.load/store/memory`は、物理型別spill array Externまたはword-packed raw memory sequenceへprovenanceに基づいてlowerする。`udonhl.suspend/frame_*`はpersistent typed spill、return state、`udonasm.jump` event return、`jump_indirect` resumeへlowerする。
- `vrc.*` → available SDK Externs + Udon heap/control-flow、またはgen C helper→udonhl。GPU Callback event stubもdata/metadataに。`vrcgpu.*` → HLSL fragments + ShaderLab + Manifest、Udon命令にはならない。
- **独自LLVM IR → High-levelライブラリ**はDialectではなくConversion/Analysis/Pass群。独自Opを強制追加しない。

## 11. GPU Layout、Texture、Manifest ABI

### 11.1 論理byteとTexture mapping

- `cl_mem` allocation of `N` bytes has immutable logical byte size `N`, zero or more versions, and physical texture configuration. `u32` ptr arithmetic strictly checks object offset on supported operations.
- RGBA8 raw encodingの**案**: 1 Texel=4 Bytes、`texel=i/4`, `lane=i%4`、`x=texel % width`, `y=texel/width`; height=`ceil(ceil(N/4)/width)`。Texelはlinear (non-sRGB), no mipmaps, no bilinear filtering, no blend、明示的round/bit operationsでbyteのbit列を保持。endian=little。これは**実機検証条件付き**で、実際のRaw formatについてShaderLab/UnityのTexture format・precision検証が必要。
- Typed Textureは`buffer_elem_type`, alignment, alias, format, physical format availability、Readback/upload結果が一致することを証明したときだけ使用する。Kernelごとにlayoutを勝手に変えず、**allocation全体・全kernelで合意**させる。
- Partial `clEnqueueWriteBuffer(offset,size)`等はbyte semantics。Shader partial writeは古いBuffer versionを入力に使って未更新byteを保持。任意書込が複数fragmentに競合する場合拒否。
- Device ABI `u32`、alignment/format/lane order、texture max size、packed bytesの反転やsRGB encodingの発生はv0テストで固定。

### 11.2 Manifest例（schema v1案）

```json
{
  "schema_version": 1,
  "compiler_version": "0.1.0-dev",
  "runtime_abi_version": 1,
  "sdk_compatibility": {"extern_db_hash": "<generated-at-build>", "sdk_version": "<validated>"},
  "target": {"host_pointer_bits": 32, "device_pointer_bits": 32, "endianness": "little", "platform": "VRChat-PC"},
  "program": {"name": "saxpy_lib", "assembly": "host.uasm"},
  "exports": [{
    "id": 1, "c_symbol": "compute_saxpy", "udon_event": "VRC_Call_compute_saxpy_1",
    "parameters": [{"name": "n", "kind": "i32", "heap_slot": "VRC_ARG_0"}],
    "result": {"kind": "i32", "heap_slot": "VRC_RET"},
    "status_slot": "VRC_STATUS", "async": true,
    "completed_event": "OnSaxpyCompleted", "required_services": ["console"]
  }],
  "device": {"kernels": [{
    "id": 7, "name": "saxpy", "shader": "kernels/saxpy.shader",
    "args": [{"index": 0, "kind": "buffer", "access": "read", "binding": "_A"},
             {"index": 1, "kind": "buffer", "access": "write", "binding": "_Out"},
             {"index": 2, "kind": "f32", "binding": "_Alpha"}],
    "passes": [{"id": 0, "shader_pass": 0, "output_arg": 1, "reads": [0], "writes": [1]}],
    "limits": {"max_bindings_profile": "<validated>"}
  }]},
  "debug": {"host_mlir": "debug/host.mlir", "device_mlir": "debug/device.mlir"}
}
```

これは**schema案**であり、実ファイルpath安全性、Unity serialization、配列要素形式、各passのversion/scratch/binding、両Host/Device export間の整合、manifest signing/cache invalidationの詳細は実装で拡張する。Version/schema migration testを必須にする。

### 11.3 OpenCL APIからManifestへ

- Runtime Cはprogram name/known source idをAOT manifestに照合。`clCreateProgramWithSource`が静的既知Kernelに対応することを**コンパイル時検証**し、実行時のソース変更は拒否または明確なbuild failure。
- `clSetKernelArg`はサイズ/型を検査して呼出時copy、pointer to `cl_mem`のvalueをHandle Tableへ保存。`clEnqueueNDRangeKernel`はrequest構築/manifest validationをし、`vrc.dispatch`に移る。
- `clFinish`とblocking read/writeは`vrc.await`からMLIR Asyncへ変換し、上流Async Runtime/Coro CFGをNative coroutineまたはUdon static-frame/trampolineへlegalizeして再開する。`cl_event`全APIは初期対象外でも内部event token/command sequenceを持ち、外部event wait listが未対応なら明示エラーにする。

## 12. 低水準U# BindingとUnity Importer

1. CompilerはMLIR export ABIからphysical Udon entry、Manifest、型付き`binding.cs`を同時生成する。
2. Bindingは`UdonBehaviour`参照を持ち、`SetProgramVariable`, `SendCustomEvent`, `GetProgramVariable`で通信する。物理event/slot名はprivate generated constantsとする。
3. 同期exportは`TryX`、suspend可能exportはreceiver/eventを受けるcallback-firstの`StartX`とstatus/result accessorを生成する。ONNX固有request/session APIは勝手に追加しない。
4. Editor Importerはcompiler artifact内のC#を配置・compileし、Program Asset/Shader/Material生成、GUID保持、参照接続、ABI validationを担当する。Manifestからbinding sourceを再生成しない。
5. `.shader` import→`Material` generation→U# script compile/domain reload→Post-import Asset/Program Bindingという複数フェーズをtransactionalに管理し、失敗時既存Assetを保持する。
6. Fingerprint、runtime/physical ABI、SDK compatibility、binding path/class、target、ready stateをImporterで検証する。Schema v1 artifactは再buildを要求する。
7. Public SDK APIだけでのcustom Udon Program Asset生成の可否を検証し、SDK private Reflectionへ黙って依存しない。
8. Versioned UPM package (`package.json`, `Editor`, `Runtime`, `Tests`, asmdef, samples)。Compiler本体はUnityにリンクしない。

## 13. テスト戦略

### 13.1 Compiler単体

- TableGen全58 Opに対してparse/print round-trip、Verifier positive/negative、invalid operands/types、Region/Terminator、symbol/format、memory effect tests。`--verify-each`、`mlir-opt`類似のPass driver。
- LLVM→MLIR recovery: scalar/loop/CFG/affine/non-affine/GEP/complex alias/bitfield/union/`__local`/OpenCL kernel metadata/conditional barrier。復元成功と**意味保持不能の明示error**の両方をGolden化。
- C ABIs: primitive sizes, `offsetof`, struct/union/bitfield/packed, pointers, pointers-int roundtrip, memcpy overlap distinction, malloc/free, UTF-8, varargs, recursion, mutual recursion, long loops。未対応libc APIは明確なfail list。
- `udonasm` 9 opcode semantics: stack source/dest order, EXTERN args/out, recursion save/return, jumps/sentinel。UdonSharp `[RecursiveMethod]`出力と照合。
- HLSL: generated ShaderLab compilation、texture format, 2D mapping, partial update, ping pong, sample or pixel center, 1D/2D/3D, private/local/global, reduction without scatter/atomics。

### 13.2 Udon VM・Unity・VRChat

- 第一にSDK付属の実際のUdon VM/Unity Test Runner。ClientSimは補助環境。確認できればUnity BatchMode/CLIテスト。独立した公式VMが存在する前提にしない。
- Shader Pass → Udon Blit → ReadbackのテストはPC VRChat**Build & Test実機**で行う。ClientSimだけではRenderTexture・callback・タイミングを最終保証しない。
- Importer tests: GUID preserved, stable wrapper class name, script reload/binding, missing program, missing SDK Extern, invalid Manifest, failed import rollback, multiple C library instances, no stale callback.
- Native OpenCL vs generated shader：小/大N、random seeds、部分転送、float32 sumの絶対相対誤差、2のべき乗work-group, multiple work-groups, replay determinism。
- U# reentrancy、async failure、GPU callback timeout/debug watchdog、Reset/Dispose、stale event、multi readback identity、Service Provider lifecycle。

### 13.3 性能・Profiling

- Direct scalar slot、固定/dynamic typed-array index、旧byte-per-element raw memory、word-packed raw memoryのload/store microbenchmarkをSDK IUdonVMで測り、opcode/extern数とwall-clockを分けて記録する。さらにarray pin/zero copy、texture upload、dispatch/passes、readback latency、total frame time、memory pressureを測り、環境/SDK/graphics API/device/texture formatをReportへ含める。
- Compeito相当手書きShaderとの比較、Udon-only、Native OpenCL。推定値を実測値と偽らない。
- Cost modelによるwarning/expensive rejectionをテストし、MRT/partial readback最適化はMVP後に評価。

## 14. 実装ディレクトリ / CMake Targets

```text
repo/
  CMakeLists.txt
  cmake/
  include/VRC/
    Dialect/UdonHL/{UdonHLDialect.td,UdonHLOps.td,UdonHLTypes.td,UdonHLAttrs.td}
    Dialect/UdonAsm/{UdonAsmDialect.td,UdonAsmOps.td,UdonAsmTypes.td,UdonAsmAttrs.td}
    Dialect/VRC/{VRCDialect.td,VRCOps.td,VRCTypes.td,VRCAttrs.td}
    Dialect/VRCGPU/{VRCGPUDialect.td,VRCGPUOps.td,VRCGPUTypes.td,VRCGPUAttrs.td}
    Conversion/LLVMToHighLevel/
    Conversion/HighLevelToUdonHL/
    Conversion/VRCToUdonHL/
    Conversion/UdonHLToUdonAsm/
    Conversion/HighLevelToVRCGPU/
    Target/UdonAssembly/
    Target/HLSL/
  lib/
    Dialect/{UdonHL,UdonAsm,VRC,VRCGPU}/
    Analysis/{PointerProvenance,OpenCLMemory,BarrierLegality,PassCost,CallGraph}/
    Conversion/{LLVMToHighLevel,HighLevelToUdonHL,VRCToUdonHL,UdonHLToUdonAsm,HighLevelToVRCGPU}/
    Target/{UdonAssembly,HLSL}/
    ABI/{Host,Device,Manifest}/
  tools/{vrccl,vrc-opt,vrc-translate}/
  runtime/c/{opencl,libc-shims,allocator,queue,handles}/
  unity/Packages/com.example.vrccl/{Editor,Runtime,Tests,Samples~,package.json}/
  tests/{dialects,recovery,codegen,abi,runtime,unity,e2e,performance}/
  examples/{scalar-lib,saxpy,reduction,libc-smoke}/
  docs/{dialects,abi,unsupported,validation}
```

**CMake library targets**（初期目標）:

- `VRCDialectUdonHL`, `VRCDialectUdonAsm`, `VRCDialectVRC`, `VRCDialectVRCGPU` — 独立TableGen出力/Op実装。
- **`VRCLLVMToHighLevelMLIR`** — **必ず独立したCMakeライブラリ**。Importer補正/metadata sidecarとanalysis/passをHost/Deviceで共用。独立unit tests。
- `VRCAnalysis`, `VRCConversionHost`, `VRCConversionDevice`, `VRCConversionVRC`, `VRCConversionUdon`, `VRCTargetUdon`, `VRCTargetHLSL`, `VRCManifest`, `VRCABIs`。
- `vrccl` (CLI build/link/generate), `vrc-opt` (Pass driver), `vrc-translate` (emit and inspect)。既存MLIR/LLVM utilityを再利用し、固定版にpinする。

## 15. Implementation Phases — Codexへの実装順序

### P0. SDK / LLVM / API feasibility spikes (**最優先、通過ゲート**)

- SDK Udon Assembly→Program Asset作成・UdonBehaviour設定のPublic API調査と最小動作。
- 実VMで9 opcode golden tests、Extern DBのSDK scraping、U# callback/variables/Custom Event、二Behavior間同期呼出。
- Fragment Shader→Blit→1x1 fence→GPU Readback順序・TryGetData、ShaderLab import・render target形式、src!=dst。失敗時はアーキテクチャ補正し、結果を記録。
- Typed Udon arrays→Texture uploadの**Q18**性能・正確性測定。代替候補（small uniform upload / Texture2D staging / chunked or other exposed APIs）と比較して採用方式を確定。CPU→GPU転送が十分でない場合、設計をそのまま進めず代替を検討。
- Clang/MLIR pinned版でOpenCL kernel metadata/import、PicolibcのClangクロスbuild、C11 ILP32 Target ABI。失敗なら修正前提を明示。
- P0完了条件：実装可能な最低API集合/SDK version/API verification logと、No-Go項目に回避策がある。

### P1. Dialects + Verifiers + Compiler plumbing

- 4 TableGen Dialectsと**58 Opすべての定義・Verifier・parse/print tests**を作成。UdonAsm 9 opcodeについてはテキストEmitterと逆Assembler goldenも。
- `VRCLLVMToHighLevelMLIR`独立ライブラリ、IR import sidecar、最小loop/CFG/memory recovery。
- Manifest schema, ABI/export symbol bindingの生成器。単体テストとCLI scaffolding。

### P2. Host scalar C → Udon library + generated low-level U# binding

- Udon Runtimeのslots, pointer ABI, basic arithmetic, flow, array/raw memory, functions, library initialization and export stubs。
- LLVM IR static link of small C runtime、export-aware whole-program optimization、`__vrc_*` substitution、非再帰interprocedural register/spill allocation、frame/continuation lowering。再帰SCCは後続機能として明示errorにする。
- Compiler-generated physical entry/Manifest/low-level `binding.cs`、Unity Asset Importer、同期export `Add`をU#から呼び出す。実Udon VMでint/f32/arrayをテストする。

### P3. GPU single-pass / Memory transfer / OpenCL API

- `cl_mem` handles, in-order Queue, Buffer allocation and byte-precise transfer, safe ping-pong, format conversion, `vrcgpu`→HLSL/ShaderLab, Manifest binding。
- `clCreateProgramWithSource/BuildProgram/CreateKernel/SetKernelArg/EnqueueNDRangeKernel/Finish`等必要APIのC Runtime、async GPU callback/continuation。
- SAXPY end-to-end。全中間ステージのIR、Udon Assembly、Shader、Cost Reportを保存。

### P3.5. General Host MLIR + public UdonBLAS Unity gate

- Legacy straight-line matcherではなく、既存`func/cf/scf/memref` recoveryからmulti-block UdonHL/UdonAsmまでを接続する。Direct call graph、固定local/global配列、typed pointer、部分buffer transferを通す。
- Blocking APIの`vrc.await`をcall graphごとMLIR Asyncへ変換し、上流`async-func-to-async-runtime`のcoroutine CFGをNativeではLLVM coroutine、Udonでは静的Heap frame + event-driven resume trampolineへlowerする。Udonにpending command schedulerやworker queueを作らない。Nativeも実event callbackでsuspend/resumeを検証する。
- Installed CMake packageから`UdonBLAS::UdonBLAS`をlinkする独立exampleを生成し、通常の`udonblasSgemm(m=1,n=10,k=144)`を9 tileすべて非ゼロのfixtureでUnity SDK IUdonVM実行する。これをONNX移行前の必須gateとする。

### P4. Reduction + barrier lowering (MVP合格条件)

- Kernel metadata、`__local`/barrier proof、local-size specializations、multiple fragment passes、inter-work-group tree sum。
- 1D/2D→3D linearization、partial write/offset、fence order tests、float32 error tests。実VRChat Client Build & Testで最終1値をU#に返す。
- 完了条件: 「U#→C library→OpenCL Sum Reduction→GPU→Readback→C/U#」とSAXPYの繰返し正確性。

### P5. libc C11 / optimization / extended tests

- Picolibc移植とU# I/O/VFS/time adapters。全ISO C11 APIごとのstatus matrix (implemented/restricted/unavailable) と例題。
- Struct/union/varargs/long double/locale/stdio/`setjmp`等の仕様と実装をABI testで拡張。大規模Host C対応、再帰dynamic frame、word-packed raw memory、register/spill/index-cache cost modelを拡張・調整する。
- Test matrix, performance regression, SDK upgrades, doc generation, UPM packaging, reproducible builds。

### P6. ONNX MLIR → library Host calls

- §2.6で固定したONNX-MLIR frontendを`udoncl-opt`/`udoncl-onnx`へ統合し、ONNXを`UdonONNXOperators`およびUdonBLAS/UdonDNN/UdonATenの公開`func.call`へloweringする。Runtime shapeは保持し、物理capacityだけをgeneric `DIMENSION_BOUNDS`で上限化する。
- 共通Device buffer/view/workspaceと汎用C Runtime session/I/O bindingを先に成立させ、C Host export + compiler-generated low-level U# bindingからMNIST、hidden-size-8 Transformer、generic `MatMulNBits`、固定TinyLlama 1.1Bの順に数値検証する。EditorはCMake directory→artifact target build/importのみを担当する。
- MNIST新経路合格後にprofile固有emitterを、generic MatMulNBits合格後に旧int4 API/kernelを、全consumer移行後にexecution-plan JSON、ONNX専用Material/arena scheduler、`static_inputs`と旧runtime/sampleを削除する。

**開発優先度**: 早くEnd-to-Endを通すことを優先するが、将来ISO C11 libc全体へ伸ばす前提でTarget ABIとmemory semanticsを最初に固定する。未対応機能を無言でスキップしない。

## 16. 検証・受入基準

| Gate | 成功の判定 | 合格前にしてはならないこと |
|---|---|---|
| G0: SDK | Program Asset生成、Udon event ABI、Readback callback、Blit順序/formatをPC実クライアントで再現 | fence保証・高速uploadを断言すること |
| G1: Dialect | 58 Op全てのTableGen定義とVerifier/parse-print、9 opcodeをSDK実VM goldenで一致 | pseudo Udon命令を追加すること |
| G2: Host | U#→Cの同期export、非再帰call、typed register/spill、word-packed raw memory、ILP32 ABI、errorsが実VMで動く | CとU#のABIを暗黙のC# object castに頼ること |
| G3: Async | `clFinish`/readbackでstateが保たれ、古いcallbackを拒否しfailure callbackを返せる | イベント間でVM call stackが残ると想定すること |
| G4: SAXPY | OpenCL C→fragment→GPU→readbackがReference結果と一致 | native Compute Shaderを使えたと主張すること |
| G5: Reduction | `__local` + barriers + multi-work-groupを変換しfloat32総和をU#に返す | unsupported scatter/atomicsをsilent compileすること |
| G6: libc | 各C11標準APIについてテスト/対応状況を明示、移植不能機能に診断 | 「完全libc対応」をテストなしに掲げること |
| G7: UdonBLAS | installed packageを`find_package`した独立exampleの`udonblasSgemm(1,10,144)`がNative async backendとUnity SDK IUdonVMでCPU referenceに一致し、9 tileのresume、reimport、cleanup、console 0を満たす | 手書きUASM、Unity専用SGEMM、source-tree-only packageで代替すること |
| G8: ONNX MLIR | 固定ONNX-MLIR frontendからexecution planなしで汎用operator/library call→共通Host/Udon binding/sessionを生成し、MNIST、合成Transformer、固定TinyLlama 1.1Bの全logits・22-layer KVがORT oracleと規定誤差内で一致 | model profile、ONNX専用Material scheduler、arena SGEMV、旧int4特殊化へfallbackすること |
| G9: Host storage / MNIST | descriptor・一時pointerの一般raw accessを除去し、allocator reportと512-address exact probeを通し、generated U#経由のUnity MNISTがnative oracleと一致して10秒watchdogを発火しない | checkpoint追加だけでbyte raw heap中心のHostを最終化すること |

## 17. 未確定・検証待ちの設計リスク（ユーザーへの未回答質問ではない）

| ID | 検証対象 | 状態 / 実装上の対策 |
|---|---|---|
| R1 / Q18 | Typed Udon Array→Staging Texture upload性能 | **条件付き未確定**。サイズ帯ごとのCPU/GPU時間を実測し、必要なら別pathへ。 |
| R2 | Fence 1×1 Blit Readback completion / GPU ordering | **未証明**。同一queue命令順だけで結論しない。stress test→必要なら別completion方式へ。 |
| R3 | Custom Program Assetを公開SDK APIで自動生成・更新 | **未証明**。P0でSDK source/public APIsと実動作検証。 |
| R4 | Udon exposed Texture2D upload API、RGBA8 byte忠実性、sRGB、RFloat、Shader Model | **未証明**。CPU readback roundtrip、幅/行境界/bit patternsテスト。 |
| R5 | 1 Behavior callback複数Readback Requestの識別 | **未証明**。識別不可ならv0 request直列化。 |
| R6 | LLVM IR→包括的High-level MLIR復元、Importer Metadata対応 | **高難度/研究実装**。sidecar+importer extension、段階的サポートとdiagnostics。 |
| R7 | Picolibc Clang/ILP32/Udon環境移植 | **要検証**。Meson buildからLLVM IRへの移植、POSIX shim、コンパイラTarget ABI。 |
| R8 | U# Wrapper zero-copyと型付き配列の同一参照共有 | **条件付き**。sharing確立しなければcopyを使い意味論優先。 |
| R9 | ShaderのテクスチャBinding上限、GPU Pass実行コスト | **Hardware/SDK-dependent**。上限値を経験で決めずManifest profileとCI testで固定。 |
| R10 | Long CPU loops/safepoint budgets・GPU callback before import init | **要実機検証**。初期化順、callback sequencing、フレーム負荷をテスト。 |
| R11 | `udonasm` extern signatures/bytecode address/layoutのSDK更新 | **SDK依存**。extern DB生成、SDK version pin、回帰テスト。 |
| R12 | HLSL Pixel Shaderでのpacked byte・uint bit operations精度 | **要検証**。符号化がbit完全でなければ代替formatを選定。 |

**原則**: これらは「後でユーザーに聞く設計判断」ではなく、Codexが**まず事実を調べてSpikeし、仕様に合う実装方法を選び、結果とtradeoffを記録するタスク**。重大な前提変更が必要になったときだけユーザーに判断を求める。

## 18. Codexに渡す最初の実装タスク

1. このPLAN.mdを読み、`docs/decisions.md`に決定事項を転記する。勝手にQ12をselective-onlyへ縮小したり、Cライブラリのentryを`main()`に変更したりしない。
2. `CMakeLists.txt`とLLVM/MLIR pinned環境の最小構成を作り、4 DialectのTableGen + 9 Udon opcodeのparsing/printing testを最初に通す。**`VRCLLVMToHighLevelMLIR`を独立CMake target**にする。
3. 同時にSDK P0 spikes用Unity Test Worldを作り、`UdonBehaviour`のProgram Asset/CustomEvent/Readbackを**実SDKとPC Clientで**検証する。ベンチ結果を`docs/validation.md`に保存。
4. 全58 OpsのVerifier/LoweringをTDDで追加し、IR dump可能なCLI `vrc-opt`を作る。未知Opを黙って通さない。
5. Sync `add` library + compiler-generated low-level U# binding → SAXPY → Reductionの順にG2〜G5を満たす。各段階にGolden tests/End-to-End testを追加。
6. Picolibc portの技術調査とC11 API coverage matrixを並行着手するが、OpenCL MVPの実装完了をlibc全面対応で遅らせない。
7. 未検証事項は`TODO: verify on target SDK/client`とTest IDを残す。「動作した」と記録するには実行ログと環境を付ける。
8. README、法的な依存ライセンス、Unity Package version pin、再現可能なbuild/test手順を維持する。

### `vrccl` CLI案

```sh
vrccl build --host src/lib.c --device kernels/saxpy.cl \
  --export=compute_saxpy --output build/saxpy --target=vrc-pc

vrc-opt --pass-pipeline='builtin.module(...)' input.mlir -o output.mlir
vrccl inspect build/saxpy/manifest.json
vrccl validate build/saxpy --sdk-extern-db path/to/externs.json
```

**コマンドラインは設計案**であり実在バイナリではない。Compilerは警告/未対応の原因とsource locationを表示し、`--dump-ir`, `--dump-after`, `--emit-cost-report`, `--allow-expensive`等を提供する。

## 19. 既存実装との関係

- **Compeito**: VRChatでFragment Shader/Material/Blitを使ったGPGPU的な手書き処理の参考。自動OpenCL→HLSL Compiler、任意scatter・barrier合法化まで提供するわけではない。
- **rvc**: Fragment ShaderでRISC-V/Linux emulationを試みた別設計の参考。Compilerの正解/直接依存ではない。
- **UdonSharp**: 実際の生成Udon Assembly、recursive method、Udon event calling convention、Unity SDK機能検証の参考。
- **Picolibc**: C11 libc移植の第一候補。OS依存関数をHost U# servicesへ接続する設計を採用。
- **MLIR**: Dialect Conversion/Operation TableGen/LLVM importerを使い、復元Converterを独立ターゲットへ。

## 20. 公式・参考リンク（2026-10-30時点、リンクの仕様が変わったら実SDKで再検証）

**VRChat / Unity**

- VRChat Udon VM & Assembly（9 opcode、Udon Assembly、heap、`.export`）: https://creators.vrchat.com/worlds/udon/vm-and-assembly/
- VRChat Udon Overview / custom compiler: https://creators.vrchat.com/worlds/udon/
- VRChat VRCGraphics / Blit: https://creators.vrchat.com/worlds/udon/vrc-graphics/
- VRChat AsyncGPUReadback / callback / TryGetData: https://creators.vrchat.com/worlds/udon/vrc-graphics/asyncgpureadback/
- VRChat ClientSim: https://creators.vrchat.com/worlds/clientsim/
- UdonSharp UdonBehaviour SendCustomEvent / SetProgramVariable: https://udonsharp.docs.vrchat.com/vrchat-api/
- Community Udon VM opcode reference: https://nori-lang.dev/internals/udon-vm/

**LLVM / MLIR**

- MLIR LLVM IR Translation / Import limitations: https://mlir.llvm.org/docs/TargetLLVMIR/
- MLIR LLVM Dialect: https://mlir.llvm.org/docs/Dialects/LLVM/
- MLIR Dialects Index: https://mlir.llvm.org/docs/Dialects/
- MLIR Operation Definition Specification (TableGen/ODS): https://mlir.llvm.org/docs/DefiningDialects/Operations/
- MLIR Defining Dialects: https://mlir.llvm.org/docs/DefiningDialects/
- MLIR GPU Dialect: https://mlir.llvm.org/docs/Dialects/GPU/

**既存プロジェクト / libc**

- Compeito: https://github.com/phi16/Compeito
- Fragment-based RISC-V emulation reference: https://github.com/pimaker/rvc
- Picolibc OS port integration: https://github.com/picolibc/picolibc/blob/main/doc/os.md
- Picolibc build documentation (Meson / Clang caveats): https://github.com/picolibc/picolibc/blob/main/doc/build.md
- Fast Udon VM（Editor代替、正しさの代替にはしない）: https://github.com/MagmaVRC/FastUdonVM

---

## 21. 最後の実装指針

- **すべての設計が実装されたと偽らない**。本PLANは仕様と実装手順であって完成レポートではない。
- ユーザーとの合意で優先されたことは、**U# entry → C library**、包括的**LLVM IR→High-level MLIR復元の独立CMake library**、**実Udon VMでの開発・テスト**、**Reduction込みMVP**、**4 Dialectと全58 Opの明示的なcatalog**。
- 最初に実機制約を測り、不可能な前提が見つかれば、それに依存する工程だけ止める。すでに確定した設計と安全性を黙って変えない。
