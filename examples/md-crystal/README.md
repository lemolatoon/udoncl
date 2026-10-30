# 触れるMD結晶 — host-native

```sh
make example-md   # コンパイル・CPU/GPU比較・保存則・局所加熱の検証
make md-view      # http://127.0.0.1:8765/ をブラウザで開く
```

ルートREADMEのLLVM 23 SDKとnative描画用依存を先に用意してください。
LinuxはLavapipe、WindowsはD3D11 WARPで実際にFragment Shaderを実行します。
ビルドは利用可能なCPU数で並列実行し、native runtime側にも
`build/md-crystal/runtime-build/compile_commands.json`を生成します。

マウスを押して結晶に触れると加熱します。タッチ入力は2点まで対応します。
加熱・計算速度、一時停止、リセットを操作できます。手を離すと加熱だけが止まり、
粒子計算は継続します。表示される運動はnativeバイナリが返す粒子状態です。
ブラウザは計算を代行しません。マウス・タッチはWorld側の手位置の代わりです。

## 計算と境界

64粒子の2次元Lennard–Jones結晶、周期境界、換算単位
`mass = sigma = epsilon = kB = 1`を使います。カットオフは2.5で、
力とポテンシャルを整合するようforce-shiftします。速度Verlet法を
`md_drift`と`md_kick`の2つのOpenCL C kernelに分割し、時間刻みは0.001です。
近い粒子対に反対向きの乱数インパルスを与え、局所的にエネルギーを注入します。
初期重心速度は0です。色は近傍・時間で平均した運動エネルギーの指標で、
熱力学的温度の厳密な局所測定ではありません。

`host.c`は通常のOpenCL 1.2 APIでバッファ作成、引数設定、NDRange、readbackを
行います。HostはLLVM → native ABIを保持するUdonHL call adapter → LLVM →
native objectの経路です。これは完全なILP32 Host memory raisingではありません。
nativeバイナリにはC製OpenCL subset runtimeと描画runtimeをリンクします。

DeviceはOpenCL C → LLVM → bounded memory recovery → `func/arith/memref/scf/gpu`
→ `vrcgpu` → HLSLです。入力・出力は別の1024バイトバッファ、
256×1 RGBA8 UNormテクスチャでfloatの4バイトを保存します。
1 invocationが1 scalar wordだけ出力し、入力からのgatherを行います。
同一入出力、未初期化の入力、範囲外、scatter、barrier、証明できないループは
この経路では受理しません。Readback・再uploadは毎dispatch行うnative検証用構成です。

`run.py`は検証コマンドを`build/md-crystal/commands.json`へ保存します。
各bundleの`frontend.recovered.mlir`と`host-device.mlir`、生成HLSL、manifest、
Hostの`host.udonhl.mlir`と`host.native.ll`を確認できます。

これは第1段階のhost-native例です。UdonAssemblyへのlowering、U#・手位置との接続、
Unity Editor拡張、VRChat実機検証は次段階です。
