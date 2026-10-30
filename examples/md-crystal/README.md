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
加熱・計算速度、一時停止、リセットを操作できます。手を離すとゼロ温度の
Langevin熱浴へ徐々に放熱し、粒子計算を続けながら液体から氷へ戻ります。
液相では運動温度から連続的に求めた液相率に応じて重力が立ち上がり、
Y方向の反射容器内を下へ流れて底に溜まります。固相では重力を無効化するため、
初期結晶の保存則には影響しません。
表示される運動はnativeバイナリが返す粒子状態です。
ブラウザは計算を代行しません。マウス・タッチはWorld側の手位置の代わりです。

## 計算と境界

32粒子の3次元FCC Lennard–Jones結晶、3軸周期境界、換算単位
`mass = sigma = epsilon = kB = 1`を使います。2×2×2周期セルの最小像面を
避けるためカットオフは1.5で、12個の第一近接粒子を含み、
力とポテンシャルを整合するようforce-shiftします。速度Verlet法を
`md_drift`と`md_kick`の2つのOpenCL C kernelに分割し、時間刻みは0.001です。
近い粒子対に反対向きの乱数インパルスを与え、局所的にエネルギーを注入します。
初期重心速度は0です。色は近傍・時間で平均した運動エネルギーの指標で、
熱力学的温度の厳密な局所測定ではありません。

冷却項は `F_f = -m v / tau` の離散化であり、Langevin熱浴の確率力を
目標温度0で省いたものです。これは自然冷却を表現する小さなモデルで、実際の
水の相図を再現するモデルではありません。相表示は3次元のSteinhardt
bond-orientational order `Q6` と平均運動エネルギーを組み合わせます。モデル化は
[LAMMPSのLangevin熱浴](https://docs.lammps.org/fix_langevin.html)の
`F_f = -m v / damp`（目標温度0では確率力が0）と、Steinhardtらの
[3D bond-orientational order](https://doi.org/10.1103/PhysRevB.28.784)
を基準にしています。液体の流動はLAMMPSの
[`fix gravity`](https://docs.lammps.org/fix_gravity.html)と
[`wall/reflect`](https://docs.lammps.org/fix_wall_reflect.html)に対応する
下向き加速度と反射境界の最小モデルです。

非interactive検証は、初期結晶を全体加熱してから放熱する決定的なcycleも実行します。
Windows D3D11 WARPで `Q6 = 0.5745 -> 0.5357 -> 0.5744`、
`T = 0.00007 -> 0.449 -> 0.00093`、平均Y座標の下降量`0.282`を観測し、
`ice -> liquid/flow -> ice` の三状態を検証しています。局所タッチは同じ加熱項を狭い
半径で使います。

`host.c`は通常のOpenCL 1.2 APIでバッファ作成、引数設定、NDRange、readbackを
行います。`device.cl`は通常のビルドでは両kernelを定義します。AOT側だけが
`MD_DRIFT` / `MD_KICK`で各entryを個別に選択します。HostはLLVM → native ABIを保持するUdonHL call adapter → LLVM →
native objectの経路です。これは完全なILP32 Host memory raisingではありません。
nativeバイナリにはC製OpenCL subset runtimeと描画runtimeをリンクします。

DeviceはOpenCL C → LLVM → bounded memory recovery → `func/arith/memref/scf/gpu`
→ `vrcgpu` → HLSLです。入力・出力は別の1024バイトバッファです。
32粒子を`x,y,z,pad,vx,vy,vz,pad`の8語に整列し、256×1 RGBA8 UNorm
テクスチャで各floatの4バイトを保存します。
1 invocationが1 scalar wordだけ出力し、入力からのgatherを行います。
同一入出力、未初期化の入力、範囲外、scatter、barrier、証明できないループは
この経路では受理しません。Readback・再uploadは毎dispatch行うnative検証用構成です。

`run.py`は検証コマンドを`build/md-crystal/commands.json`へ保存します。
各bundleの`frontend.recovered.mlir`と`host-device.mlir`、生成HLSL、manifest、
Hostの`host.udonhl.mlir`と`host.native.ll`を確認できます。

これはnative oracleであり、同じHost/Device契約をUdon Assembly、Unity Worldの
手位置入力と表示へ移すときの比較基準です。
