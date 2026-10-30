# PLAN.md — VRChat向け C / OpenCL C → Udon / HLSL コンパイラ

> Status: **Design agreed / P0–P1 bootstrap in progress (gates not passed)**  
> Design cut-off: **2026-10-30**
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
  └─ generated typed U# Wrapper  [Unity Editor Importer generates]
       └─ generated C library UdonBehaviour (per-instance globals / heap / call stack)
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
   continuation/       barrier elimination / cost-aware phase splitting
   recursion lowering     │
         │              vrcgpu (fragment-safe passes)
      udonasm              │
         │             direct HLSL + ShaderLab
      .uasm                │
         └───────────┬─────┘
                     ▼
           manifest.json + debug/dumps
                     ▼
    Unity Editor Importer (UPM package)
    - Udon Program Asset / Behaviour binding
    - Shader / Material / Texture configs
    - typed UdonSharp Wrapper generation
    - asset GUID preservation + ABI validation
                     ▼
         VRChat PC World (Build & Test)
```

**Compiler/Unity責務**: CLIが生成するのは`.uasm`, `.shader`, `manifest.json`, diagnostics/debug dumpsであり、Unityの`.asset`の内部形式には依存しない。Editor Importerが`.asset`、Material、U# wrapper C#、Bindingを生成する。Wrapper自体をMLIRで生成しない。

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
- Host C pointerは32-bit flat byte virtual address。SSA scalarはHeap Slot。確実なprovenanceを持つ配列はNative Udon typed arrayへ直接lowering、混合alias/byte再解釈はallocation単位でraw linear memoryへ戻す。Cのポインタ/構造体/unionの意味論を保つ。
- 再帰（相互再帰含む）はUdonSharp `[RecursiveMethod]`相当の手動stack save/restore（call graph SCC解析）。`udonasm`に擬似CALLは追加しない。UdonSharp出力AssemblyのGolden Testで参照。
- 1 library instance = 1 Host Execution Context; 1 in-order OpenCL Command Queue。複数pending GPU requestsを管理できる構造にするがrequest識別が不確実ならMVPは直列化。別C呼び出しがbusy中はBusy拒否。完了callbackはbusy解除後にU#へ通知し、再呼び出し可。
- `cl_mem`は32-bit handle table。`clSetKernelArg`の引数は**設定時にコピー**。OpenCL 1.2 AOT互換サブセット。`clCreateProgramWithSource`は静的に既知のKernel Manifestを解決し、`clBuildProgram`は既存Assetと対応付けるだけ。実行時の任意OpenCL C source生成は未対応。
- OpenCL APIの契約・エラー処理は**C Runtime**。Udon/VRChat GPU操作・Service呼出・同期は`__vrc_*` compiler intrinsics → `vrc`へ。Host + Runtime LLVM IRを静的リンク。
- libcは**ISO C11標準ライブラリ全体を対応目標**とし、Picolibcを第一候補として移植。環境依存部分（console/VFS/time等）は必要ならU# Serviceで実装し、Cと接続。初期VFSはinstanceごとのin-memory、U#サービス側への委譲も可。stdin bufferにデータがなければ初期はEOF、stdinでの非同期待機は後日。stdout/stderrはC側buffer→一括U#通知。
- `malloc`/`free`は容量上限・失敗時`NULL`、ライフサイクルは`Initialize/Reset/Dispose`。環境依存C関数に実装不可能な処理があれば静かに偽装せず、ドキュメント化し、適切なエラー/制約を返す。
- Safepointにより長いCPUループを複数フレームに分割（最初は明示yield、以後自動Budget）。GPUの非同期callbackと同様にResume stateが生存する。

### 2.3 U# API、Unity、デバッグ

- **アプリのエントリーポイントはユーザーのUdonSharp**。`main()`自動実行ではない。Cはリンク可能なライブラリ。Export属性（例`annotate("vrc_export")`）を付けたC関数のみ公開。
- ExportごとにManifestから**型付きU# wrapper**をUnity Editor Importerが生成。同期呼出：予約Heap変数へのSetProgramVariable →公開Custom Event → GetProgramVariable。非同期：Start +完了通知Event/Status。Custom Event名は公開仕様に則り**先頭に`_`を付けない**安定ハッシュ名にする。
- 同期/非同期はCall Graph解析で自動判定し、GPU待機可能性が呼出元まで伝播。U#には`async/await`を要求しない。
- 文字列はUTF-8 null終端C byte buffer ↔ U# stringをコピー。基本型・Udon typed arraysは直接、構造体はWrapperマーシャリング、生ポインタをpublic APIに出さない。共有配列は同一Udon array objectを参照できると実機検証されたときだけzero-copy、実行中の参照/寿命条件を明示。
- 1 UdonBehaviour instanceごとにGlobals/Heap/Stack/VFS/サービスBindingが独立。`Initialize()`, `Reset()`, `Dispose()`とgeneration counterで古いGPU callbackを無視。未完了commandのTextureは安全になるまで保持。
- 非同期失敗も完了Eventで通知＋Status（成功と区別）。Debug watchdogは診断のみで誤判定を避ける。
- Unity統合は**外部Clang/MLIR CLI + Unity Editor Importer**、UPM package。Shader/Material/Udon Program Asset生成、wrapper C#生成とscript reload後のBindingの2段階処理。GUID保持・トランザクション的Import（失敗時既存正常Asset維持）。ManifestにCompiler/Runtime/SDK/ABI versionを記録し検証。
- Udon ExternデータベースはインストールSDKの公開情報から生成/検証。`udonasm`は実opcode 1:1、TableGenで4 Dialectのop/type/attrを定義し、各々独立CMake library。**LLVM IR→High-level MLIR復元はそれ自体が独立CMake library**。
- 開発時は、利用可能な**SDK付属Udon VM**を第一に利用。Unity Test Runner / ClientSim、実VRChat Build & Test。スタンドアロンVMが実証できれば活用するが、独自VMやFastUdonVMのみで正しさを断言しない。Compiler自体はmacOS/Linux上でUnityなしでもbuild/test可。

## 3. ABIとExecution Contract

### 3.1 Host C ABI

- Target `vrc-udon-unknown`（仮名）。整数のサイズ、bitfield layout、struct packing、long double、`va_list`, `setjmp`などは**Clang TargetInfo/DataLayoutでテストベースに確定**すること。`int/long/pointer=32`, little-endianは決定済みであり、host tripleを偽の既存ターゲットに紛れ込ませない。
- Heapのbyte addressing、effective type、strict aliasing、volatile、`memcpy`, unaligned access、signed/unsigned conversionをC仕様に沿って実装。native arraysは最適化手段でありABIの公開型ではない。
- メモリ領域の割当表は`base: u32`, `size: u32`, `kind: raw|typed`, `generation`, `backing`, `permissions`を持つ。`ptrtoint`/`inttoptr`後のprovenanceが曖昧ならallocation lookup + byte-oriented semanticsへ。uintptr overflow/invalid memory accessはDebug診断と必要なRelease安全チェック。
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

`Clang → LLVM IR (host + C Runtime + libc static link) → Import → Common High-level Recovery → ABI/alias analysis → udonhl (with func/scf/cf/arith) → vrc intrinsic substitution → SCC recursion lowering → Continuation liveness and frame lowering → typed-array/raw-memory selection → safepoint insertion → Udon extern legalization → udonasm → verifier → .uasm + export ABI manifest`。

順序上、非同期Call Graphとlive-value analysisは再帰stack frame layout・slot allocationと連携し、**GPU callbackで従来のUdon event stackは残らない**ことを前提にする。Static slot/inliningは普通のcall向けに最適化する。

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

## 6. `udonhl` Dialect — 全Operation一覧（22 Op）

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

**補足**: `udonhl`には独自`add/mul/if/for/return/branch/function definition`を作らない。`udonhl.trap`も初期Opではなく、必要な診断/停止はRuntime service + `cf` loweringで表現する。U# Export entry stubは`func.func`の`vrc.export_id`, `udon_event`等の属性で示す。`array_view`の一貫したC byte semanticsを証明できない場合はzero-copyしない。

### `udonhl`の重要不変条件

1. `udonhl.load/store`は完全なC byte memory modelに従う。Typed native arrayは最後のloweringの最適化であり、strict aliasing未証明を理由に結果を変えない。
2. 全`udonhl.suspend`は対応するresume stateを持ち、live scalarと再帰stackを保存済み。永続化不可能なUdon object handleを持ち越すケースはVerifierで拒否。
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
- Udon VMにnative function call/return/localsはない。High-levelのcallは`JUMP` + 戻り先`SystemUInt32` + `JUMP_INDIRECT` + Heap保存にLowering。相互再帰はSCCでstack framesを構成する。
- 値計算・配列操作はSDK Externを使い、`PUSH` sequence→`EXTERN`へ。`udonasm.extern`は**単一のVM命令**であり、引数PUSHを隠すHigh-level pseudo-opにはしない。
- `.data` declarations: name / Udon type（例`SystemInt32Array`）/ initial value / visibility/sync mode。Static Slot Allocator・Global Init・Export ABI stubが生成する。labelsとcode addressesはAssembler/relocation passが付ける。`udonasm`はSSA型の値レジスタを導入しない。
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

- 独自Dialect v0.1の完全一覧: `udonhl` **22**, `udonasm` **9**, `vrc` **13**, `vrcgpu` **13**、合計**57 Op**。この文書で明記しない独自Opは**初期仕様に存在しない**。新規追加は設計改訂＋対応するlowering/Verifier/Testを必須とする。
- 標準MLIR Op全体を57に数えない（`func`, `arith`, `math`, `cf`, `scf`, `gpu`, `memref`, `affine`, `llvm`, 必要なら`vector`等）。自前再定義はしない。Importer/Raising中の標準Dialect OpについてもLegalization/Passごとのallowlistを設ける。
- `udonhl.call` + `func.func` → Udon static call slots / recursive frame code → `udonasm.push`, `udonasm.copy`, `udonasm.jump`, `udonasm.jump_indirect`、data/metaはExport directiveへ。
- `udonhl.load/store/memory` → typed array Externまたはraw memory Extern/IR sequences（provenanceに基づく）。`udonhl.suspend/frame_*` → heap slots / return state / `udonasm.jump` event returnと`jump_indirect` resume等。
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
- `clFinish`とblocking read/writeの継続は`vrc.await`→`udonhl.suspend`で再開。`cl_event`全APIは初期対象外でも内部event token/command sequenceを持つ。外部event wait listは未対応なら明示エラー。

## 12. U# WrapperとUnity Importer

1. CompilerはExport signature、Heap slots、イベント名、sync/async、Service requirementsとGPU pass metadataをManifestへ。
2. Editor Importerの独立したC#ライブラリ **`VRCWrapperGenerator`** がManifestからU# C#クラスを生成。`Editor`の本体はProgram Asset/Shader/Material生成とGUID保持、binding、validationを担当。
3. U# Wrapperは`UdonBehaviour`参照を持ち、`SetProgramVariable`, `SendCustomEvent`, `GetProgramVariable`で通信。UdonSharpで実際に認められたvalue type、array、event名か、最小End-to-Endで検証。
4. 同期メソッドは戻り値型とStatusを保証。Asyncメソッドは`StartX(args)`＋`OnXCompleted()`、`GetXResult()`＋`GetXStatus()`など明確なAPIを生成。U#側がイベントを明示登録できるようにする。
5. `Initialize/Reset/Dispose`、Busy、Generation、failed readback、reentrant callbackの振る舞いをGeneratorとRuntimeで一致させる。
6. `.shader` import→`Material` generation→U# script compile/domain reload→Post-import Asset/Program Bindingという複数フェーズ。中間状態をLibrary/Editor metadataで管理し、失敗時既存Assetを保持。Asset pathの安定性、GUID/scene referenceの保持をテスト。
7. Public SDK APIだけでのcustom Udon Program Asset生成の可否を検証。公開APIで不可能なら**最初の実装Spikeで修正方針を決める**。SDK private Reflectionへの黙った依存は禁止。
8. Versioned UPM package (`package.json`, `Editor`, `Runtime`, `Tests`, asmdef, samples)。Compiler本体はUnityにリンクしない。

## 13. テスト戦略

### 13.1 Compiler単体

- TableGen全57 Opに対してparse/print round-trip、Verifier positive/negative、invalid operands/types、Region/Terminator、symbol/format、memory effect tests。`--verify-each`、`mlir-opt`類似のPass driver。
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

- Per-operation CPU Udon execution cost, native typed array vs raw byte memory, array pin/zero copy, texture upload, dispatch/passes, readback latency, total frame time、memory pressure。Reportに環境/SDK/graphics API/device/texture formatを記録。
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

- 4 TableGen Dialectsと**57 Opすべての定義・Verifier・parse/print tests**を作成。UdonAsm 9 opcodeについてはテキストEmitterと逆Assembler goldenも。
- `VRCLLVMToHighLevelMLIR`独立ライブラリ、IR import sidecar、最小loop/CFG/memory recovery。
- Manifest schema, ABI/export symbol bindingの生成器。単体テストとCLI scaffolding。

### P2. Host scalar C → Udon library + generated U# Wrapper

- Udon Runtimeのslots, pointer ABI, basic arithmetic, flow, array/raw memory, functions, library initialization and export stubs。
- LLVM IR static link of small C runtime, `__vrc_*` substitution、SCC recursion、frame/continuation lowering。
- `VRCWrapperGenerator`、Unity Asset Importer、同期export `Add`をU#から呼び出す。実Udon VMでint/f32/arrayをテスト。

### P3. GPU single-pass / Memory transfer / OpenCL API

- `cl_mem` handles, in-order Queue, Buffer allocation and byte-precise transfer, safe ping-pong, format conversion, `vrcgpu`→HLSL/ShaderLab, Manifest binding。
- `clCreateProgramWithSource/BuildProgram/CreateKernel/SetKernelArg/EnqueueNDRangeKernel/Finish`等必要APIのC Runtime、async GPU callback/continuation。
- SAXPY end-to-end。全中間ステージのIR、Udon Assembly、Shader、Cost Reportを保存。

### P4. Reduction + barrier lowering (MVP合格条件)

- Kernel metadata、`__local`/barrier proof、local-size specializations、multiple fragment passes、inter-work-group tree sum。
- 1D/2D→3D linearization、partial write/offset、fence order tests、float32 error tests。実VRChat Client Build & Testで最終1値をU#に返す。
- 完了条件: 「U#→C library→OpenCL Sum Reduction→GPU→Readback→C/U#」とSAXPYの繰返し正確性。

### P5. libc C11 / optimization / extended tests

- Picolibc移植とU# I/O/VFS/time adapters。全ISO C11 APIごとのstatus matrix (implemented/restricted/unavailable) と例題。
- Struct/union/varargs/long double/locale/stdio/`setjmp`等の仕様と実装をABI testで拡張。大規模Host C対応、provenance+typed arrays最適化、frame budgeting、Cost model tuning。
- Test matrix, performance regression, SDK upgrades, doc generation, UPM packaging, reproducible builds。

**開発優先度**: 早くEnd-to-Endを通すことを優先するが、将来ISO C11 libc全体へ伸ばす前提でTarget ABIとmemory semanticsを最初に固定する。未対応機能を無言でスキップしない。

## 16. 検証・受入基準

| Gate | 成功の判定 | 合格前にしてはならないこと |
|---|---|---|
| G0: SDK | Program Asset生成、Udon event ABI、Readback callback、Blit順序/formatをPC実クライアントで再現 | fence保証・高速uploadを断言すること |
| G1: Dialect | 57 Op全てのTableGen定義とVerifier/parse-print、9 opcodeをSDK実VM goldenで一致 | pseudo Udon命令を追加すること |
| G2: Host | U#→Cの同期export、再帰、typed/raw memory、ILP32 ABI、errorsが実VMで動く | CとU#のABIを暗黙のC# object castに頼ること |
| G3: Async | `clFinish`/readbackでstateが保たれ、古いcallbackを拒否しfailure callbackを返せる | イベント間でVM call stackが残ると想定すること |
| G4: SAXPY | OpenCL C→fragment→GPU→readbackがReference結果と一致 | native Compute Shaderを使えたと主張すること |
| G5: Reduction | `__local` + barriers + multi-work-groupを変換しfloat32総和をU#に返す | unsupported scatter/atomicsをsilent compileすること |
| G6: libc | 各C11標準APIについてテスト/対応状況を明示、移植不能機能に診断 | 「完全libc対応」をテストなしに掲げること |

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
4. 全57 OpsのVerifier/LoweringをTDDで追加し、IR dump可能なCLI `vrc-opt`を作る。未知Opを黙って通さない。
5. Sync `add` library + generated U# Wrapper → SAXPY → Reductionの順にG2〜G5を満たす。各段階にGolden tests/End-to-End testを追加。
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
- ユーザーとの合意で優先されたことは、**U# entry → C library**、包括的**LLVM IR→High-level MLIR復元の独立CMake library**、**実Udon VMでの開発・テスト**、**Reduction込みMVP**、**4 Dialectと全57 Opの明示的なcatalog**。
- 最初に実機制約を測り、不可能な前提が見つかれば、それに依存する工程だけ止める。すでに確定した設計と安全性を黙って変えない。
